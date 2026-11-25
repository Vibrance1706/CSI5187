from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from tfidf_model import find_best_threshold
from tfidf_ablation import evaluate_predictions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR = (RESULTS_DIR / "supplementary" / "hybrid")
SPLIT_TYPES = ["stratified", "project_grouped"]
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
C_OPTIONS = [0.01, 0.1, 1.0, 10.0, 100.0]
CLASS_WEIGHTS = [None, "balanced"]
SEMANTIC_WEIGHTS = [0.5, 1.0, 2.0]
RANDOM_STATE = 42
BATCH_SIZE = 32


def generate_features(texts, vectorizer, embedding_model):

    # Transform using previously fitted TF-IDF vectorizers.
    lexical_features = vectorizer.transform(texts)
    lexical_features = normalize(lexical_features,norm="l2",axis=1).tocsr()

    # Generate frozen MPNet embeddings.

    semantic_features = embedding_model.encode(
        texts.tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    semantic_features = normalize(semantic_features,norm="l2", axis=1)
    assert semantic_features.shape[1] == 768
    assert np.isfinite(semantic_features).all()
    assert np.isfinite(lexical_features.data).all()
    assert (lexical_features.shape[0] == semantic_features.shape[0] == len(texts))
    
    return lexical_features, csr_matrix(semantic_features)


def combine_features(lexical, semantic, semantic_weight):

    return hstack([lexical, semantic * semantic_weight], format="csr")


def positive_probabilities(classifier, X):

    positive_index = list(classifier.classes_).index(1)

    return classifier.predict_proba(X)[:, positive_index]


def run_hybrid(split_type, embedding_model):

    print(f"\n{'=' * 65}")
    print(f"Hybrid classification: {split_type}")
    print(f"{'=' * 65}")

    split_dir = DATA_DIR / split_type
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "validation.csv")
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    # Reuse the original fitted TF-IDF feature extractor.

    original_tfidf = joblib.load( RESULTS_DIR / "models" / split_type/ "tfidf_logistic_regression.joblib")
    
    vectorizer = original_tfidf.named_steps["features"]
    print("\nGenerating training features...")
    
    lexical_train, semantic_train = generate_features(train_df["bodyText"],vectorizer,embedding_model)
    print("\nGenerating validation features...")
    
    lexical_val, semantic_val = generate_features(val_df["bodyText"],vectorizer,embedding_model)
    print("Lexical training shape:", lexical_train.shape)
    print("Semantic training shape:", semantic_train.shape)

    best_score = -1.0
    best_classifier = None
    best_config = None
    search_results = []

    # Hyperparameter selection using validation data only.

    for semantic_weight in SEMANTIC_WEIGHTS:

        X_train = combine_features(lexical_train,semantic_train,semantic_weight )
        X_val = combine_features(lexical_val, semantic_val,semantic_weight)
        for C in C_OPTIONS:
            for class_weight in CLASS_WEIGHTS:
                classifier = LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
                classifier.fit(X_train, y_train)
                val_predictions = classifier.predict(X_val)
                score = f1_score(y_val,val_predictions,average="macro",zero_division=0)
                config = {
                    "semantic_weight": semantic_weight,
                    "C": C,
                    "class_weight": class_weight
                }
                search_results.append({ "split": split_type,**config,"validation_macro_f1": score})
                print(f"Alpha={semantic_weight:<4} | "f"C={C:<6} | "f"weight={str(class_weight):<8} | "f"Validation F1={score:.4f}")

                if score > best_score + 1e-12:
                    best_score = score
                    best_classifier = classifier
                    best_config = config.copy()

    assert best_classifier is not None

    # Optimize threshold for the selected classifier.

    X_val = combine_features(lexical_val,semantic_val,best_config["semantic_weight"])
    val_probabilities = positive_probabilities(best_classifier, X_val)
    threshold, optimized_f1, default_f1 = (find_best_threshold(y_val, val_probabilities))

    print("\nSelected hybrid configuration:")
    print(best_config)
    print(f"Default validation F1: {default_f1:.6f}")
    print(f"Selected threshold: {threshold:.6f}")
    print(f"Optimized validation F1: {optimized_f1:.6f}")

    # Model selection is complete.
    # Load the test partition for final evaluation.

    test_df = pd.read_csv(split_dir / "test.csv")
    y_test = test_df["label"].to_numpy()
    print("\nGenerating test features...")
    lexical_test, semantic_test = generate_features(test_df["bodyText"],vectorizer,embedding_model)
    X_test = combine_features(lexical_test, semantic_test,best_config["semantic_weight"])
    print("Hybrid test shape:", X_test.shape)
    test_probabilities = positive_probabilities( best_classifier, X_test )
    test_predictions = (test_probabilities >= threshold).astype(int)
    metrics = evaluate_predictions( y_test, test_predictions)

    result = {
        "split": split_type,
        "model": "hybrid",
        **best_config,
        "lexical_dimension": int(lexical_train.shape[1]),
        "semantic_dimension": int(semantic_train.shape[1]),
        "hybrid_dimension": int(X_test.shape[1]),
        "decision_threshold": float(threshold),
        "default_validation_macro_f1": float(default_f1),
        "optimized_validation_macro_f1": float(optimized_f1),
        **metrics
    }

    output_dir = OUTPUT_DIR / split_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the complete fitted hybrid model components.

    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": best_classifier,
        "embedding_model": MODEL_NAME,
        "max_seq_length": embedding_model.max_seq_length,
        "embedding_dimension": 768,
        "block_normalization": "l2",
        "normalize_embeddings": False,
        "semantic_weight": best_config["semantic_weight"],
        "decision_threshold": float(threshold)
    }

    joblib.dump(model_bundle,output_dir / "hybrid_logistic_regression.joblib")
    
    with open( output_dir / "hybrid_metrics.json","w",encoding="utf-8") as file:
        json.dump(result,file,indent=4, allow_nan=False)

    prediction_df = test_df.copy()
    prediction_df["probability"] = test_probabilities
    prediction_df["prediction"] = test_predictions

    prediction_df["correct"] = ( test_predictions == y_test )
    prediction_df.to_csv(output_dir / "hybrid_predictions.csv", index=False)

    # Compare against the unchanged original classifiers.

    comparison = []
    for model_type in ["tfidf", "semantic"]:

        reference = pd.read_csv( RESULTS_DIR / "predictions" / split_type / f"{model_type}_predictions.csv")
        assert np.array_equal(
            reference["id"].to_numpy(),
            test_df["id"].to_numpy()
        )

        assert np.array_equal( reference["label"].to_numpy(), y_test)
        reference_predictions = ( reference["prediction"].to_numpy())
        comparison.append({"split": split_type,"model": model_type, **evaluate_predictions( y_test, reference_predictions)})

    comparison.append({"split": split_type, "model": "hybrid", **metrics})

    print("\nHybrid test performance:")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    print(f"Positive recall: {metrics['positive_recall']:.6f}")

    return result, comparison, search_results


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    embedding_model = SentenceTransformer(MODEL_NAME,device=device )
    print("Embedding dimension:", embedding_model.get_embedding_dimension())
    print("Maximum sequence length:",embedding_model.max_seq_length)
    assert embedding_model.get_embedding_dimension() == 768
    assert embedding_model.max_seq_length == 384
    all_results = []
    all_comparisons = []
    all_search_results = []

    for split_type in SPLIT_TYPES:

        result, comparison, search_results = run_hybrid(split_type,embedding_model)
        all_results.append(result)
        all_comparisons.extend(comparison)
        all_search_results.extend(search_results)

    results_df = pd.DataFrame(all_results)
    comparison_df = pd.DataFrame(all_comparisons)
    search_df = pd.DataFrame(all_search_results)

    assert len(search_df) == 60
    results_df.to_csv(OUTPUT_DIR / "hybrid_results.csv",index=False)
    comparison_df.to_csv(OUTPUT_DIR / "hybrid_comparison.csv",index=False)
    search_df.to_csv( OUTPUT_DIR / "hybrid_validation_search.csv",index=False)
    print("\nFINAL HYBRID CONFIGURATIONS")
    print(results_df[["split","semantic_weight","C","class_weight","hybrid_dimension","decision_threshold","optimized_validation_macro_f1"]].round(6).to_string(index=False))
    print("\nFINAL MODEL COMPARISON")
    print(comparison_df[["split", "model","accuracy", "macro_precision", "macro_recall","macro_f1", "positive_precision","positive_recall","positive_f1"]].round(6).to_string(index=False))
    print("\nValidation configurations:", len(search_df))
    print("Results saved to:", OUTPUT_DIR)

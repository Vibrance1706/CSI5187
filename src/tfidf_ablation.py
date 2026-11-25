from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score,recall_score, f1_score,confusion_matrix)
from tfidf_model import find_best_threshold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR = (RESULTS_DIR / "supplementary" / "tfidf_ablation")
SPLIT_TYPES = ["stratified", "project_grouped"]
C_OPTIONS = [0.1, 1.0, 10.0, 100.0]
CLASS_WEIGHTS = [None, "balanced"]


def positive_probabilities(model, X):
    
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, positive_index]


def evaluate_predictions(y_true, predictions):
    
    tn, fp, fn, tp = confusion_matrix(y_true,predictions,labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_precision": precision_score( y_true, predictions,average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, predictions,average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, predictions,average="macro", zero_division=0),
        "positive_precision": precision_score(y_true, predictions, zero_division=0),
        "positive_recall": recall_score( y_true, predictions, zero_division=0),
        "positive_f1": f1_score(y_true, predictions, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def run_ablation(split_type):
    
    print(f"\n{'=' * 65}")
    print(f"TF-IDF ablation: {split_type}")
    print(f"{'=' * 65}")
    split_dir = DATA_DIR / split_type
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "validation.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    X_train = train_df["bodyText"]
    y_train = train_df["label"]
    X_val = val_df["bodyText"]
    y_val = val_df["label"]
    X_test = test_df["bodyText"]
    y_test = test_df["label"]
    output_dir = OUTPUT_DIR / split_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the existing combined TF-IDF classifier

    original_model = joblib.load(RESULTS_DIR / "models" / split_type/ "tfidf_logistic_regression.joblib")

    with open(RESULTS_DIR / "metrics" / split_type/ "tfidf_validation.json", encoding="utf-8") as file:
        original_config = json.load(file)

    original_classifier = (original_model.named_steps["classifier"])
    original_vectorizers = dict(original_model.named_steps["features"].transformer_list)
    assert set(original_vectorizers) == {"word", "char"}
    original_C = original_classifier.C
    original_weight = original_classifier.class_weight
    original_threshold = float(original_config["best_threshold"])
    print("Original C:", original_C)
    print("Original class weight:", original_weight)
    selected_models = {
        "word_char": {
            "model": original_model,
            "threshold": original_threshold,
            "fixed_model": original_model
        }
    }

    search_results = []

    # Train and select the two ablated models.
    # No test predictions are used during model selection.

    for feature_type in ["word", "char"]:
        variant = f"{feature_type}_only"
        print(f"\nFeature configuration: {variant}")
        best_score = -1.0
        best_model = None
        fixed_model = None

        for C in C_OPTIONS:
            for class_weight in CLASS_WEIGHTS:
                vectorizer = clone(original_vectorizers[feature_type])
                classifier = clone(original_classifier).set_params( C=C,class_weight=class_weight)
                model = Pipeline([("features", vectorizer),("classifier", classifier)])
                model.fit(X_train, y_train)
                val_predictions = model.predict(X_val)
                score = f1_score( y_val,val_predictions,average="macro",zero_division=0)
                search_results.append({
                    "split": split_type,
                    "variant": variant,
                    "C": C,
                    "class_weight": class_weight,
                    "validation_macro_f1": score
                })

                print(f"C={C:<6} | " f"weight={str(class_weight):<8} | " f"Validation F1={score:.4f}")

                # Retain the model using the original
                # classifier hyperparameters.

                if (C == original_C and class_weight == original_weight):
                    fixed_model = model

                # Select using validation macro-F1.
                if score > best_score + 1e-12:
                    best_score = score
                    best_model = model

        assert best_model is not None
        assert fixed_model is not None

        val_probabilities = positive_probabilities(best_model, X_val)
        threshold, optimized_f1, default_f1 = (find_best_threshold(y_val,val_probabilities))
        selected_models[variant] = {"model": best_model,"threshold": float(threshold),"fixed_model": fixed_model}
        print(f"\nSelected configuration: {variant}")
        print("Best C:",best_model.named_steps["classifier"].C)
        print("Class weight:",best_model.named_steps["classifier"].class_weight)
        print(f"Selected threshold: {threshold:.6f}")
        print(f"Optimized validation F1: {optimized_f1:.6f}")
        joblib.dump( best_model,output_dir / f"{variant}_logistic_regression.joblib")

    # All model configurations and thresholds are now fixed.
    # Evaluate the test partition.

    results = []
    controlled_results = []
    prediction_df = test_df.copy()

    for variant in ["word_only", "char_only", "word_char"]:
        config = selected_models[variant]
        model = config["model"]
        threshold = config["threshold"]
        fixed_model = config["fixed_model"]
        classifier = model.named_steps["classifier"]
        val_probabilities = positive_probabilities( model, X_val)
        test_probabilities = positive_probabilities( model, X_test)
        default_val_f1 = f1_score( y_val,(val_probabilities >= 0.5).astype(int),average="macro")
        optimized_val_f1 = f1_score(y_val,(val_probabilities >= threshold).astype(int),average="macro")
        predictions = (test_probabilities >= threshold).astype(int)
        metrics = evaluate_predictions( y_test, predictions)

        # Verify the unchanged combined model against
        # the original saved test predictions.

        if variant == "word_char":
            reference = pd.read_csv(RESULTS_DIR / "predictions" / split_type/ "tfidf_predictions.csv")
            assert np.array_equal(reference["id"].to_numpy(), test_df["id"].to_numpy())
            assert np.array_equal(reference["label"].to_numpy(),y_test.to_numpy())
            assert np.array_equal( reference["prediction"].to_numpy(),predictions), "Original combined predictions differ."
            assert np.isclose(optimized_val_f1,original_config["validation_macro_f1"])

        result = {
            "split": split_type,
            "variant": variant,
            "C": classifier.C,
            "class_weight": classifier.class_weight,
            "decision_threshold": threshold,
            "default_validation_macro_f1": default_val_f1,
            "optimized_validation_macro_f1": optimized_val_f1,
            **metrics
        }
        results.append(result)

        prediction_df[f"{variant}_probability"] = (test_probabilities)
        prediction_df[f"{variant}_prediction"] = ( predictions)
        prediction_df[f"{variant}_correct"] = (predictions == y_test.to_numpy())

        # Controlled ablation using the original
        # classifier hyperparameters and threshold 0.5.

        fixed_val_probabilities = positive_probabilities(fixed_model, X_val)
        fixed_test_probabilities = positive_probabilities(fixed_model, X_test)
        fixed_predictions = (fixed_test_probabilities >= 0.5).astype(int)
        controlled_results.append({
            "split": split_type,
            "variant": variant,
            "C": original_C,
            "class_weight": original_weight,
            "decision_threshold": 0.5,
            "validation_macro_f1": f1_score(y_val,(fixed_val_probabilities >= 0.5).astype(int),average="macro"),
            **evaluate_predictions(
                y_test, fixed_predictions
            )
        })

        with open(output_dir / f"{variant}_metrics.json", "w",encoding="utf-8") as file:
            json.dump(result, file,indent=4,allow_nan=False)

    prediction_df.to_csv(output_dir / "ablation_predictions.csv",index=False)

    return results, controlled_results, search_results


if __name__ == "__main__":

    all_results = []
    all_controlled_results = []
    all_search_results = []

    for split_type in SPLIT_TYPES:
        results, controlled, search = run_ablation(split_type)
        all_results.extend(results)
        all_controlled_results.extend(controlled)
        all_search_results.extend(search)

    results_df = pd.DataFrame(all_results)
    controlled_df = pd.DataFrame( all_controlled_results)
    search_df = pd.DataFrame(all_search_results)
    results_df.to_csv(OUTPUT_DIR / "ablation_comparison.csv",index=False)
    controlled_df.to_csv(OUTPUT_DIR / "ablation_controlled_comparison.csv",index=False)
    search_df.to_csv(OUTPUT_DIR / "ablation_validation_search.csv",index=False)
    print("\nFINAL OPTIMIZED ABLATION RESULTS")
    print(results_df[["split","variant","C","class_weight","decision_threshold","optimized_validation_macro_f1","accuracy","macro_f1","positive_precision","positive_recall" ]].round(6).to_string(index=False))
    print("\nCONTROLLED ABLATION RESULTS")
    print(controlled_df[["split","variant","C", "class_weight","validation_macro_f1","accuracy","macro_f1"]].round(6).to_string(index=False))
    print("\nValidation configurations:", len(search_df))
    print("Results saved to:", OUTPUT_DIR)

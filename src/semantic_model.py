
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_TYPES = ["stratified", "project_grouped"]

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# C_OPTIONS = [0.1, 1.0, 10.0] - v1
C_OPTIONS = [0.01, 0.1, 1.0, 10.0, 100.0] 
# NORMALIZE_OPTIONS = [False, True] -- normalize didn't work
SCALING_OPTIONS = [False, True]
CLASS_WEIGHTS = [None, "balanced"]
RANDOM_STATE = 42
BATCH_SIZE = 32


def generate_embeddings(model, texts):
    """Generate fixed semantic embeddings for comments."""

    embeddings = model.encode(texts.tolist(),batch_size=BATCH_SIZE,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=False)
    return embeddings

def find_best_threshold(y_true, probabilities):

    unique_probs = np.unique(probabilities)

    # Test decision boundaries between observed
    # validation probabilities.
    midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], midpoints)))
    # Start with the standard 0.5 threshold
    best_threshold = 0.5
    best_f1 = f1_score(y_true,(probabilities >= 0.5).astype(int), average="macro")
    default_f1 = best_f1

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score( y_true,predictions,average="macro")

        if score > best_f1 + 1e-12:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1, default_f1

def train_semantic(split_type, embedding_model):

    print(f"\n{'=' * 50}")
    print(f"Experiment: {split_type}")
    print(f"{'=' * 50}")

    # Step 1: Load the appropriate partitions
    split_dir = DATA_DIR / split_type

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "validation.csv")

    X_train = train_df["bodyText"]
    y_train = train_df["label"]

    X_val = val_df["bodyText"]
    y_val = val_df["label"]


    print("\nGenerating training embeddings...")
    train_embeddings = generate_embeddings(embedding_model, X_train)

    print("\nGenerating validation embeddings...")
    val_embeddings = generate_embeddings(embedding_model, X_val)

    print("Training embedding shape:", train_embeddings.shape)
    print("Validation embedding shape:", val_embeddings.shape)

    # best_score = -1 -v1
    # best_classifier = None
    # best_C = None
    # for C in C_OPTIONS:

    #     classifier = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)

    #     classifier.fit(train_embeddings, y_train)

    #     predictions = classifier.predict(val_embeddings)

    #     macro_f1 = f1_score(y_val, predictions, average="macro")

    #     print(f"C: {C}, "f"Validation Macro-F1: {macro_f1:.4f}")

    #     if macro_f1 > best_score:
    #         best_score = macro_f1
    #         best_classifier = classifier
    #         best_C = C

    # best_predictions = best_classifier.predict(val_embeddings)
    
#     best_score = -1 -- normalize didn't work
#     best_classifier = None
#     best_C = None
#     best_normalize_embeddings = None
#     best_val_features = None

# # Compare raw embeddings against L2-normalized embeddings
#     for use_normalization in NORMALIZE_OPTIONS:

#         if use_normalization:
#             train_features = normalize(train_embeddings, norm="l2")
#             val_features = normalize(val_embeddings, norm="l2")
#         else:
#             train_features = train_embeddings
#             val_features = val_embeddings

#         for C in C_OPTIONS:

#             classifier = LogisticRegression(C=C,class_weight="balanced",max_iter=2000,random_state=RANDOM_STATE)

#             classifier.fit(train_features, y_train)
#             predictions = classifier.predict(val_features)
#             macro_f1 = f1_score(y_val,predictions,average="macro")
#             print(f"Normalize: {use_normalization}, "f"C: {C}, "f"Validation Macro-F1: {macro_f1:.4f}")

#             if macro_f1 > best_score:
#                 best_score = macro_f1
#                 best_classifier = classifier
#                 best_C = C
#                 best_normalize_embeddings = use_normalization
#                 best_val_features = val_features


#     best_predictions = best_classifier.predict(best_val_features)


    best_score = -1
    best_classifier = None
    best_C = None
    best_scaling = None
    best_class_weight = None

# Compare original MPNet embeddings with standardized embeddings
    for use_scaling in SCALING_OPTIONS:
        for C in C_OPTIONS:
            for class_weight in CLASS_WEIGHTS:
                classifier = Pipeline([
                    ("scaler", 
                     StandardScaler() if use_scaling else "passthrough"
                    ),
                    ("classifier",
                    LogisticRegression(
                        C=C,
                        class_weight=class_weight,
                        max_iter=2000,
                        random_state=RANDOM_STATE
                    )   
                    )
                ])

        # Fit the scaler (if enabled) and classifier
        # using training embeddings only.
                classifier.fit(train_embeddings, y_train)
                predictions = classifier.predict(val_embeddings)
                macro_f1 = f1_score(y_val,predictions,average="macro")
                print(f"StandardScaler: {use_scaling}, "f"C: {C}, " f"Validation Macro-F1: {macro_f1:.4f}")

                if macro_f1 > best_score:
                    best_score = macro_f1
                    best_classifier = classifier
                    best_C = C
                    best_scaling = use_scaling
                    best_class_weight = class_weight

# Generate predictions using the selected pipeline
    # best_predictions = best_classifier.predict(val_embeddings)
    
# Generate validation probabilities
    val_probabilities = best_classifier.predict_proba(val_embeddings    )[:, list(best_classifier.classes_).index(1)]

# Automatically identify the optimal threshold
    best_threshold, best_score, default_f1 = find_best_threshold(y_val, val_probabilities)

    best_predictions = (val_probabilities >= best_threshold).astype(int)

    print(f"\nOptimal threshold: {best_threshold:.4f}")
    print(f"Default threshold Macro-F1: {default_f1:.4f}")
    print(f"Optimized threshold Macro-F1: {best_score:.4f}")
    print(f"Best class weight: {best_class_weight}")
    


    # print(f"\nBest C: {best_C}") -v1
    # print(f"Best validation Macro-F1: {best_score:.4f}")
    # print(f"\nBest C: {best_C}") - normalize didn't work
    # print(f"Best embedding normalization: {best_normalize_embeddings}")
    # print(f"Best validation Macro-F1: {best_score:.4f}")
    print(f"\nBest C: {best_C}")
    print(f"Best StandardScaler setting: {best_scaling}")
    print(f"Best validation Macro-F1: {best_score:.4f}")

    print("\nValidation Classification Report:")
    print(classification_report(y_val,best_predictions, target_names=["No", "Yes"], zero_division=0))

    model_dir = RESULTS_DIR / "models" / split_type
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_classifier,model_dir / "semantic_logistic_regression.joblib")

    metrics_dir = RESULTS_DIR / "metrics" / split_type
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "split_type": split_type,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": int(train_embeddings.shape[1]),
        "max_seq_length": embedding_model.max_seq_length,
        "normalize_embeddings": False,
        "standardize_embeddings": best_scaling,
        "best_C": best_C,
        "class_weight": best_class_weight,
        "best_threshold": float(best_threshold),
        "default_threshold_validation_macro_f1": float(default_f1),
        "validation_macro_f1": float(best_score)
    }

    with open(metrics_dir / "semantic_validation.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nClassifier and validation metrics saved.")

    return best_classifier, results


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    embedding_model = SentenceTransformer(MODEL_NAME, device=device)
    print("Embedding dimension:",embedding_model.get_embedding_dimension())
    print("Maximum sequence length:",embedding_model.max_seq_length)
    for split_type in SPLIT_TYPES:
        train_semantic(split_type, embedding_model)

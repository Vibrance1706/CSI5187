
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_TYPES = ["stratified", "project_grouped"]

# NGRAM_OPTIONS = [(1, 1), (1, 2)] - v1
C_OPTIONS = [0.1, 1.0, 10.0, 100.0]
CLASS_WEIGHTS = [None, "balanced"]
FEATURE_CONFIGS = [
    {
        "word_ngrams": (1, 2),
        "char_ngrams": (2, 5),
        "min_df": 2
    },
    {
        "word_ngrams": (1, 3),
        "char_ngrams": (2, 5),
        "min_df": 2
    },
    {
        "word_ngrams": (1, 2),
        "char_ngrams": (3, 5),
        "min_df": 2
    }
]

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

def train_tfidf(split_type):

    print(f"\n{'=' * 50}")
    print(f"Experiment: {split_type}")
    print(f"{'=' * 50}")

    split_dir = DATA_DIR / split_type

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "validation.csv")

    X_train = train_df["bodyText"]
    y_train = train_df["label"]

    X_val = val_df["bodyText"]
    y_val = val_df["label"]

    best_score = -1
    best_model = None
    best_params = None

    # Hyperparameter selection using validation data - v1
    # for ngram_range in NGRAM_OPTIONS:
    #     for C in C_OPTIONS:

    #         model = Pipeline([
    #             ("tfidf", TfidfVectorizer(ngram_range=ngram_range, lowercase=True)),
    #             ("classifier", LogisticRegression(C=C,class_weight="balanced",max_iter=2000,random_state=42))
    #             ])

    #         model.fit(X_train, y_train)

    #         predictions = model.predict(X_val)

    #         macro_f1 = f1_score(y_val,predictions,average="macro")

    #         print(f"N-grams: {ngram_range}, "f"C: {C}, "f"Validation Macro-F1: {macro_f1:.4f}")

    #         if macro_f1 > best_score:
    #             best_score = macro_f1
    #             best_model = model
    #             best_params = {
    #                 "ngram_range": list(ngram_range),
    #                 "C": C
    #             }
    
    # Hyperparameter selection using validation data
    for config in FEATURE_CONFIGS:
        for C in C_OPTIONS:
            for class_weight in CLASS_WEIGHTS:

                model = Pipeline([
                ("features", FeatureUnion([
                    ("word", TfidfVectorizer(
                        analyzer="word",
                        ngram_range=config["word_ngrams"],
                        token_pattern=r"(?u)\b\w+\b",
                        min_df=config["min_df"],
                        sublinear_tf=True
                    )),
                    ("char", TfidfVectorizer(
                        analyzer="char",
                        ngram_range=config["char_ngrams"],
                        min_df=config["min_df"],
                        max_features=30000,
                        sublinear_tf=True
                    ))
                ])),
                ("classifier", LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42
                ))
            ])

            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            macro_f1 = f1_score(y_val, predictions, average="macro")
            print(f"Word n-grams: {config['word_ngrams']}, " f"Char n-grams: {config['char_ngrams']}, " f"C: {C}, " f"Class weight: {class_weight}, " f"Validation Macro-F1: {macro_f1:.4f}")

            if macro_f1 > best_score:
                best_score = macro_f1
                best_model = model
                best_params = {
                    "word_ngram_range": list(config["word_ngrams"]),
                    "char_ngram_range": list(config["char_ngrams"]),
                    "min_df": config["min_df"],
                    "sublinear_tf": True,
                    "C": C,
                    "class_weight": class_weight
                }

    # best_predictions = best_model.predict(X_val)
    
# Generate validation probabilities
    val_probabilities = best_model.predict_proba(X_val)[:, list(best_model.classes_).index(1)]


# Automatically identify the optimal threshold
    best_threshold, best_score, default_f1 = find_best_threshold(y_val,val_probabilities)
    best_predictions = (val_probabilities >= best_threshold).astype(int)

    print(f"\nOptimal threshold: {best_threshold:.4f}")
    print(f"Default threshold Macro-F1: {default_f1:.4f}")
    print(f"Optimized threshold Macro-F1: {best_score:.4f}")

    print("\nBest hyperparameters:", best_params)
    print(f"Best validation Macro-F1: {best_score:.4f}")

    print("\nValidation Classification Report:")
    print(classification_report(y_val, best_predictions, target_names=["No", "Yes"], zero_division=0))


    model_dir = RESULTS_DIR / "models" / split_type
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_dir / "tfidf_logistic_regression.joblib")

    metrics_dir = RESULTS_DIR / "metrics" / split_type
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "split_type": split_type,
        "best_hyperparameters": best_params,
        "validation_macro_f1": best_score,
        "best_threshold": float(best_threshold),
    "default_threshold_validation_macro_f1": float(default_f1),
    "validation_macro_f1": float(best_score)
    }

    with open(metrics_dir / "tfidf_validation.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nModel and validation results saved.")

    return best_model, results


if __name__ == "__main__":

    for split_type in SPLIT_TYPES:
        train_tfidf(split_type)

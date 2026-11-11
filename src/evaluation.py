
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix,ConfusionMatrixDisplay)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_TYPES = ["stratified", "project_grouped"]
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def evaluate_predictions(y_true, y_pred):
    """Calculate classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro",zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred,average="macro",zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], target_names=["No", "Yes"], output_dict=True, zero_division=0)
    }


def save_confusion_matrix(y_true, y_pred, split_type, model_type):

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_type} - {split_type}")
    output_dir = RESULTS_DIR / "figures" / split_type
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{model_type}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model(split_type, model_type, embedding_model=None):

    print(f"\n{'=' * 55}")
    print(f"Evaluating {model_type} | {split_type}")
    print(f"{'=' * 55}")

    test_df = pd.read_csv(DATA_DIR / split_type / "test.csv")

    X_test = test_df["bodyText"]
    y_test = test_df["label"]

    model_dir = RESULTS_DIR / "models" / split_type

    # if model_type == "tfidf":
    #     model = joblib.load(model_dir / "tfidf_logistic_regression.joblib")
    #     predictions = model.predict(X_test)

    # elif model_type == "semantic":
    #     classifier = joblib.load(model_dir / "semantic_logistic_regression.joblib")
    #     embeddings = embedding_model.encode(X_test.tolist(),batch_size=32,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=False)
    #     predictions = classifier.predict(embeddings)

    # else:
    #     raise ValueError(f"Unknown model type: {model_type}")
    
# Load the threshold selected using validation data
    config_path = (RESULTS_DIR / "metrics" / split_type /f"{model_type}_validation.json")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if "best_threshold" not in config:
        raise ValueError(f"Optimized threshold missing from {config_path}")

    best_threshold = float(config["best_threshold"])

    if not 0.0 <= best_threshold <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")

# Load model and prepare the appropriate features
    if model_type == "tfidf":
        classifier = joblib.load(model_dir / "tfidf_logistic_regression.joblib")
        model_input = X_test

    elif model_type == "semantic":
        classifier = joblib.load(model_dir / "semantic_logistic_regression.joblib")
        model_input = embedding_model.encode(
        X_test.tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=config.get(
            "normalize_embeddings", False
            )
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Obtain probabilities for the positive class (Yes = 1)
    positive_class_index = list(classifier.classes_).index(1)
    probabilities = classifier.predict_proba(model_input)[:, positive_class_index]
# Apply the validation-selected decision threshold
    predictions = (probabilities >= best_threshold).astype(int)

    print(f"Selected decision threshold: {best_threshold:.4f}")


    metrics = evaluate_predictions(y_test, predictions)
    metrics["decision_threshold"] = best_threshold

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro-Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test,predictions,labels=[0, 1],target_names=["No", "Yes"],zero_division=0))

    save_confusion_matrix(y_test, predictions,split_type, model_type)

    prediction_dir = (RESULTS_DIR / "predictions" / split_type)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = test_df.copy()
    prediction_df["prediction"] = predictions
    prediction_df["correct"] = (prediction_df["label"] == prediction_df["prediction"])

    prediction_df.to_csv(prediction_dir / f"{model_type}_predictions.csv", index=False)


    metrics_dir = RESULTS_DIR / "metrics" / split_type
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / f"{model_type}_test.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nResults saved successfully.")

    return {
        "split": split_type,
        "model": model_type,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"]
    }


if __name__ == "__main__":

    embedding_model = SentenceTransformer(MODEL_NAME)

    all_results = []

    for split_type in SPLIT_TYPES:

        for model_type in ["tfidf", "semantic"]:

            result = evaluate_model(split_type, model_type, embedding_model)

            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    print("\nFINAL TEST PERFORMANCE")
    print(results_df.to_string(index=False))

    results_df.to_csv(RESULTS_DIR / "metrics" / "test_comparison.csv", index=False)

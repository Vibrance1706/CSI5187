
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

SPLIT_TYPES = ["stratified", "project_grouped"]


def analyze_errors(split_type):

    print(f"\n{'=' * 55}")
    print(f"Error Analysis: {split_type}")
    print(f"{'=' * 55}")

    split_dir = PREDICTIONS_DIR / split_type

    tfidf_df = pd.read_csv(split_dir / "tfidf_predictions.csv")

    semantic_df = pd.read_csv(split_dir / "semantic_predictions.csv")

    comparison = tfidf_df[["id", "project", "bodyText", "label", "prediction"]].rename(columns={"prediction": "tfidf_prediction"})

    semantic_predictions = semantic_df[["id", "label", "prediction"]].rename(columns={"label": "semantic_label","prediction": "semantic_prediction"})

    comparison = comparison.merge(semantic_predictions,on="id",how="inner",validate="one_to_one")


    assert len(comparison) == len(tfidf_df) == len(semantic_df)
    assert (comparison["label"] == comparison["semantic_label"]).all()

    comparison = comparison.drop(columns=["semantic_label"])


    comparison["tfidf_correct"] = (comparison["tfidf_prediction"] == comparison["label"])

    comparison["semantic_correct"] = (comparison["semantic_prediction"] == comparison["label"])

    comparison["disagreement"] = (comparison["tfidf_prediction"]!= comparison["semantic_prediction"])

    for model in ["tfidf", "semantic"]:

        prediction_col = f"{model}_prediction"

        false_positives = comparison[(comparison["label"] == 0) & (comparison[prediction_col] == 1)]

        false_negatives = comparison[(comparison["label"] == 1) & (comparison[prediction_col] == 0)]

        print(f"\n{model.upper()}")
        print(f"False positives: {len(false_positives)}")
        print(f"False negatives: {len(false_negatives)}")

    disagreements = comparison[comparison["disagreement"]].copy()

    tfidf_only_correct = comparison[comparison["tfidf_correct"] & ~comparison["semantic_correct"]]
    semantic_only_correct = comparison[comparison["semantic_correct"] & ~comparison["tfidf_correct"]]

    both_wrong = comparison[ ~comparison["tfidf_correct"] & ~comparison["semantic_correct"] ]

    print("\nMODEL COMPARISON")
    print(f"Total test comments: {len(comparison)}")
    print(f"Model disagreements: {len(disagreements)}")
    print(f"Only TF-IDF correct: {len(tfidf_only_correct)}")
    print(f"Only MPNet correct: {len(semantic_only_correct)}")
    print(f"Both models incorrect: {len(both_wrong)}")

    output_dir = PROJECT_ROOT / "results" / "error_analysis" / split_type
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_dir / "all_predictions.csv",index=False)

    disagreements.to_csv(output_dir / "model_disagreements.csv", index=False)

    tfidf_only_correct.to_csv(output_dir / "tfidf_only_correct.csv",index=False)

    semantic_only_correct.to_csv(output_dir / "semantic_only_correct.csv", index=False)

    both_wrong.to_csv(output_dir / "both_models_incorrect.csv", index=False)


    print("\nSample disagreements:")

    columns = ["bodyText","label","tfidf_prediction","semantic_prediction"]

    with pd.option_context("display.max_colwidth", 150):
        print(disagreements[columns].head(10).to_string(index=False))

    print("\nError analysis files saved successfully.")


if __name__ == "__main__":

    for split_type in SPLIT_TYPES:
        analyze_errors(split_type)

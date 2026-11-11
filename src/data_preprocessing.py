
from pathlib import Path
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_cleaned.csv"

TEXT_COLUMN = "bodyText"
TARGET_COLUMN = "RQ1 - Understandability?"

LABEL_MAPPING = {"No": 0,"Yes": 1}


def preprocess_dataset():

   
    df = pd.read_csv(RAW_DATA_PATH, sep="\t")

    print(f"Original dataset size: {len(df)}")

    
    required_columns = ["project","id",TEXT_COLUMN,TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    
    df = df[df[TARGET_COLUMN].isin(["Yes", "No"])].copy()

    
    if df[TEXT_COLUMN].isna().any():
        raise ValueError("Missing comment text detected.")

    if df[TEXT_COLUMN].str.strip().eq("").any():
        raise ValueError("Empty comment text detected.")

    
    df["label"] = df[TARGET_COLUMN].map(LABEL_MAPPING)

    
    df = df[["project","id",TEXT_COLUMN,"label"]].copy()

    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Processed dataset size: {len(df)}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print(f"\nProcessed dataset saved to:\n{OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    preprocess_dataset()

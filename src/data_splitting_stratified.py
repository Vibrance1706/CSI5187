
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split



# Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (PROJECT_ROOT / "data" / "processed" / "dataset_cleaned.csv")

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stratified"

RANDOM_STATE = 42


def check_overlap(train_df, val_df, test_df, column):
    """Check for values shared across data partitions."""

    train_values = set(train_df[column])
    val_values = set(val_df[column])
    test_values = set(test_df[column])

    train_val = train_values & val_values
    train_test = train_values & test_values
    val_test = val_values & test_values

    print(f"\nOverlap analysis for '{column}':")
    print(f"Train / Validation: {len(train_val)}")
    print(f"Train / Test:       {len(train_test)}")
    print(f"Validation / Test: {len(val_test)}")


def split_dataset():
    """Create reproducible stratified 70/15/15 partitions."""

    df = pd.read_csv(INPUT_PATH)

    print(f"Total records: {len(df)}")

    required_columns = ["project", "id", "bodyText", "label"]

    if not set(required_columns).issubset(df.columns):
        raise ValueError("Required columns are missing.")

    if df[required_columns].isna().any().any():
        raise ValueError("Missing values detected.")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Unexpected classification labels.")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate record IDs detected.")

    train_val_df, test_df = train_test_split(df,test_size=0.15, stratify=df["label"], random_state=RANDOM_STATE)


    train_df, val_df = train_test_split(train_val_df,test_size=15 / 85,stratify=train_val_df["label"],random_state=RANDOM_STATE)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)


    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert train_ids | val_ids | test_ids == set(df["id"])

    for name, subset in [("Training", train_df), ("Validation", val_df), ("Testing", test_df)]:
        print(f"\n{name} set: {len(subset)} records")
        print("Class distribution:")
        print(subset["label"].value_counts().sort_index())

        print("Class percentages:")
        print(subset["label"].value_counts(normalize=True).sort_index().mul(100).round(2))

    check_overlap(train_df, val_df, test_df,column="bodyText")

    check_overlap(train_df, val_df, test_df,column="project")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nAll dataset partitions saved successfully.")


if __name__ == "__main__":
    split_dataset()

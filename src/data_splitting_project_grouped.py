
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (PROJECT_ROOT / "data" / "processed" / "dataset_cleaned.csv")

OUTPUT_DIR = (PROJECT_ROOT / "data" / "processed" / "project_grouped")

RANDOM_STATE = 42
N_CANDIDATES = 400


def find_best_group_split(
    df, group_fraction, target_fraction, total_records, seed
):
    """
    Generate group-disjoint candidate splits and select
    the one closest to the desired size and label ratio.
    """

    splitter = GroupShuffleSplit(n_splits=N_CANDIDATES,test_size=group_fraction,random_state=seed)

    target_size = total_records * target_fraction
    target_positive_rate = df["label"].mean()

    best_score = float("inf")
    best_indices = None

    for remaining_idx, heldout_idx in splitter.split(df,y=df["label"],groups=df["project"]):
        heldout = df.iloc[heldout_idx]

        size_difference = (abs(len(heldout) - target_size) / total_records)

        class_difference = abs(heldout["label"].mean() - target_positive_rate)

        score = size_difference + 2 * class_difference

        if score < best_score:
            best_score = score
            best_indices = (remaining_idx, heldout_idx)

    remaining_df = df.iloc[best_indices[0]].copy()
    heldout_df = df.iloc[best_indices[1]].copy()

    return remaining_df, heldout_df


def check_overlap(train_df, val_df, test_df, column):
    """Count distinct shared values across partitions."""

    train_values = set(train_df[column])
    val_values = set(val_df[column])
    test_values = set(test_df[column])

    print(f"\nOverlap analysis for '{column}':")
    print("Train / Validation:", len(train_values & val_values))
    print("Train / Test:      ", len(train_values & test_values))
    print("Validation / Test:", len(val_values & test_values))


def split_dataset():

    df = pd.read_csv(INPUT_PATH)

    required = {"project", "id", "bodyText", "label"}

    if not required.issubset(df.columns):
        raise ValueError("Required columns are missing.")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values detected.")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Invalid target labels.")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate record IDs detected.")

    total_records = len(df)
    print(f"Total records: {total_records}")
    print(f"Total projects: {df['project'].nunique()}")

    train_val_df, test_df = find_best_group_split(df=df,group_fraction=0.15,target_fraction=0.15, total_records=total_records, seed=RANDOM_STATE)


    train_df, val_df = find_best_group_split(df=train_val_df, group_fraction=15 / 85,target_fraction=0.15,total_records=total_records, seed=RANDOM_STATE + 1)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    assert len(train_df) + len(val_df) + len(test_df) == total_records

    for column in ["id", "project"]:
        partitions = [set(part[column])for part in [train_df, val_df, test_df]]

        assert partitions[0].isdisjoint(partitions[1])
        assert partitions[0].isdisjoint(partitions[2])
        assert partitions[1].isdisjoint(partitions[2])

    assert (set(train_df["id"]) | set(val_df["id"]) | set(test_df["id"])) == set(df["id"])


    for name, subset in [("Training", train_df), ("Validation", val_df), ("Testing", test_df)]:
        print(f"\n{name} set: {len(subset)} records")
        print(f"Projects: {subset['project'].nunique()}")

        print("Class distribution:")
        print(subset["label"].value_counts().sort_index())

        print("Class percentages:")
        print(subset["label"].value_counts(normalize=True).sort_index().mul(100).round(2))

    check_overlap(train_df, val_df, test_df, "id")
    check_overlap(train_df, val_df, test_df, "project")
    check_overlap(train_df, val_df, test_df, "bodyText")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nProject-grouped partitions saved successfully.")


if __name__ == "__main__":
    split_dataset()

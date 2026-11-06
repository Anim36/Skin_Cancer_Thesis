import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

base = Path("reports/dataset_inspection")
source = base / "master_dataset_manifest_clean.csv"

df = pd.read_csv(source, dtype=str).fillna("").reset_index(drop=True)

required = ["image_id", "canonical_label", "split_group_id", "sha256"]
for col in required:
    if col not in df.columns:
        raise SystemExit(f"Missing required column: {col}")

for col in required:
    df[col] = df[col].str.strip()

if df[required].eq("").any().any():
    raise SystemExit("Empty required values found. Stop.")

# Every group must have exactly one class
group_label_counts = df.groupby("split_group_id")["canonical_label"].nunique()
if (group_label_counts != 1).any():
    raise SystemExit("Mixed-label groups found. Stop.")

# Every exact image hash must be unique
if df["sha256"].duplicated().any():
    raise SystemExit("Duplicate SHA256 found. Stop.")

y = df["canonical_label"].to_numpy()
groups = df["split_group_id"].to_numpy()
X = np.zeros((len(df), 1))

# Stage 1: 7 folds for train, 3 folds for temporary validation/test
outer = StratifiedGroupKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

fold_ids = np.full(len(df), -1, dtype=int)

for fold, (_, held_out_idx) in enumerate(outer.split(X, y, groups)):
    fold_ids[held_out_idx] = fold

if (fold_ids < 0).any():
    raise SystemExit("Some records were not assigned to an outer fold.")

train_mask = fold_ids >= 3
temp_mask = fold_ids < 3

train_df = df.loc[train_mask].copy()
temp_df = df.loc[temp_mask].copy()

# Stage 2: split temporary 30% into validation and test
inner = StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=42
)

temp_y = temp_df["canonical_label"].to_numpy()
temp_groups = temp_df["split_group_id"].to_numpy()
temp_X = np.zeros((len(temp_df), 1))

val_idx, test_idx = next(inner.split(temp_X, temp_y, temp_groups))

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

result = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

# Leakage checks
sets = {
    name: set(part["split_group_id"])
    for name, part in result.groupby("split")
}

sha_sets = {
    name: set(part["sha256"])
    for name, part in result.groupby("split")
}

for a, b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test")
]:
    if sets[a] & sets[b]:
        raise SystemExit(f"Group leakage detected: {a} vs {b}")
    if sha_sets[a] & sha_sets[b]:
        raise SystemExit(f"SHA256 leakage detected: {a} vs {b}")

# Check all records assigned exactly once
if len(result) != len(df):
    raise SystemExit("Record count mismatch.")

if result["image_id"].duplicated().any():
    raise SystemExit("Duplicate image IDs found.")

# Save split manifest
output = base / "master_dataset_manifest_split.csv"
result.to_csv(output, index=False)

# Summary
print("=" * 70)
print("GROUP-STRATIFIED DATASET SPLIT SUMMARY")
print("=" * 70)

print("\nOverall split distribution:")
print(
    result.groupby("split").agg(
        images=("image_id", "size"),
        unique_groups=("split_group_id", "nunique")
    ).to_string()
)

print("\nSplit percentages:")
print(
    (result["split"].value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)

print("\nClass distribution by split:")
class_table = pd.crosstab(
    result["canonical_label"],
    result["split"]
).reindex(columns=["train", "validation", "test"], fill_value=0)

print(class_table.to_string())

print("\nClass percentages within each split:")
class_percent = (
    class_table.div(class_table.sum(axis=0), axis=1) * 100
).round(2)
print(class_percent.to_string())

print("\nDataset distribution by split:")
print(pd.crosstab(result["dataset"], result["split"]).to_string())

print("\nLeakage checks:")
print("Train/Validation group overlap:", len(sets["train"] & sets["validation"]))
print("Train/Test group overlap:", len(sets["train"] & sets["test"]))
print("Validation/Test group overlap:", len(sets["validation"] & sets["test"]))
print("Train/Validation SHA overlap:", len(sha_sets["train"] & sha_sets["validation"]))
print("Train/Test SHA overlap:", len(sha_sets["train"] & sha_sets["test"]))
print("Validation/Test SHA overlap:", len(sha_sets["validation"] & sha_sets["test"]))

print("\nClass presence in every split:")
for split_name, part in result.groupby("split"):
    missing = sorted(set(y) - set(part["canonical_label"]))
    print(split_name, "missing classes:", missing)

print("\nSaved:", output)
print("\nSPLIT BUILD COMPLETE.")

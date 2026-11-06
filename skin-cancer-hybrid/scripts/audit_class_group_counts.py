import pandas as pd
from pathlib import Path

path = Path("reports/dataset_inspection/master_dataset_manifest_clean.csv")
df = pd.read_csv(path, dtype=str).fillna("")

df["canonical_label"] = df["canonical_label"].str.strip()
df["split_group_id"] = df["split_group_id"].str.strip()

# Each group must contain only one class
group_labels = df.groupby("split_group_id")["canonical_label"].nunique()

if (group_labels > 1).any():
    raise SystemExit("ERROR: Mixed-label split groups found.")

# Count images and unique groups for every class
summary = df.groupby("canonical_label").agg(
    image_count=("image_id", "size"),
    unique_groups=("split_group_id", "nunique")
).reset_index()

summary["images_per_group"] = (
    summary["image_count"] / summary["unique_groups"]
).round(2)

print("=" * 65)
print("CLASS-WISE LESION GROUP AUDIT")
print("=" * 65)
print(summary.to_string(index=False))

print("\nTotal images:", len(df))
print("Total unique groups:", df["split_group_id"].nunique())
print("Groups with mixed labels:", int((group_labels > 1).sum()))

summary.to_csv(
    "reports/dataset_inspection/class_group_summary.csv",
    index=False
)

print("\nSaved: reports/dataset_inspection/class_group_summary.csv")

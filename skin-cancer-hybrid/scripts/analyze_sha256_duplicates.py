import pandas as pd
from pathlib import Path
from itertools import combinations

p = Path("reports/dataset_inspection/image_sha256_inventory.csv")
df = pd.read_csv(p)

groups = (
    df.groupby("sha256")
    .agg(
        copies=("sha256", "size"),
        dataset_count=("dataset", "nunique"),
        datasets=("dataset", lambda x: ";".join(sorted(set(x)))),
        names=("image_name", lambda x: ";".join(x))
    )
    .reset_index()
)

cross = groups[groups["dataset_count"] > 1].copy()
within = groups[
    (groups["dataset_count"] == 1) & (groups["copies"] > 1)
].copy()

print("DUPLICATE GROUP BREAKDOWN")
print("=" * 55)

print("\nCross-dataset groups by exact dataset combination:")
print(cross.groupby("datasets").size().sort_values(ascending=False).to_string())

print("\nCross-dataset duplicate groups by number of datasets:")
print(cross.groupby("dataset_count").size().to_string())

print("\nWithin-dataset duplicate groups:")
print(within.groupby("datasets").size().to_string())

print("\nPAD-UFES20 cross-dataset duplicate groups:")
pad = cross[cross["datasets"].str.contains("PAD_UFES20")]
print("Count:", len(pad))
if len(pad):
    print(pad[["datasets", "copies", "names"]].head(20).to_string(index=False))

print("\nDuplicate copy count:")
print("Total files:", len(df))
print("Unique hashes:", df["sha256"].nunique())
print("Redundant copies:", len(df) - df["sha256"].nunique())

print("\nDataset-wise exact duplicate file copies:")
for dataset in sorted(df["dataset"].unique()):
    sub = df[df["dataset"] == dataset]
    print(
        f"{dataset}: files={len(sub)}, "
        f"unique_hashes={sub['sha256'].nunique()}, "
        f"redundant_copies={len(sub)-sub['sha256'].nunique()}"
    )

out = Path("reports/dataset_inspection")
cross.to_csv(out / "cross_dataset_duplicate_groups.csv", index=False)
within.to_csv(out / "within_dataset_duplicate_groups.csv", index=False)

print("\nSaved detailed duplicate group reports.")

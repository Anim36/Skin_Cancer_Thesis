import pandas as pd
from pathlib import Path

source = Path("reports/dataset_inspection/master_dataset_manifest_sha_fixed.csv")
out_dir = Path("reports/dataset_inspection")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source, dtype=str)

required = [
    "dataset", "image_id", "lesion_id", "canonical_label",
    "sha256", "split_group_id", "split_eligible"
]

missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Missing columns: {missing_cols}")

for col in ["dataset", "image_id", "canonical_label", "sha256", "split_group_id"]:
    df[col] = df[col].fillna("").str.strip()

print("=" * 65)
print("DATASET LEAKAGE AND GROUP INTEGRITY AUDIT")
print("=" * 65)

# 1. Duplicate image IDs within each dataset
id_dups = df[df.duplicated(["dataset", "image_id"], keep=False)].copy()
id_dups.to_csv(out_dir / "audit_duplicate_image_ids.csv", index=False)

print("\n1. Duplicate dataset + image IDs:", len(id_dups))

# 2. Exact SHA256 duplicate groups
valid_sha = df[df["sha256"].str.len() == 64].copy()
sha_counts = valid_sha.groupby("sha256").agg(
    image_count=("image_id", "size"),
    dataset_count=("dataset", "nunique"),
    group_count=("split_group_id", "nunique"),
    datasets=("dataset", lambda x: "|".join(sorted(set(x)))),
    image_ids=("image_id", lambda x: "|".join(sorted(set(x)))),
    split_groups=("split_group_id", lambda x: "|".join(sorted(set(x))))
).reset_index()

duplicate_sha = sha_counts[sha_counts["image_count"] > 1].copy()
cross_dataset_sha = duplicate_sha[duplicate_sha["dataset_count"] > 1].copy()
cross_group_sha = duplicate_sha[duplicate_sha["group_count"] > 1].copy()

duplicate_sha.to_csv(out_dir / "audit_exact_duplicate_sha_groups.csv", index=False)
cross_dataset_sha.to_csv(out_dir / "audit_cross_dataset_sha_groups.csv", index=False)
cross_group_sha.to_csv(out_dir / "audit_cross_split_group_sha_groups.csv", index=False)

print("\n2. Valid SHA256 rows:", len(valid_sha))
print("Unique SHA256 hashes:", valid_sha["sha256"].nunique())
print("Exact duplicate SHA groups:", len(duplicate_sha))
print("Cross-dataset duplicate SHA groups:", len(cross_dataset_sha))
print("Duplicates spanning multiple split groups:", len(cross_group_sha))

# 3. Lesion/split group label consistency
group_stats = df.groupby("split_group_id").agg(
    image_count=("image_id", "size"),
    label_count=("canonical_label", "nunique"),
    labels=("canonical_label", lambda x: "|".join(sorted(set(x)))),
    dataset_count=("dataset", "nunique")
).reset_index()

mixed_groups = group_stats[group_stats["label_count"] > 1].copy()
mixed_groups.to_csv(out_dir / "audit_mixed_label_split_groups.csv", index=False)

print("\n3. Unique split groups:", len(group_stats))
print("Groups containing multiple labels:", len(mixed_groups))

# 4. Check empty SHA or split group values
print("\n4. Missing SHA values:", int((df["sha256"] == "").sum()))
print("Missing split group values:", int((df["split_group_id"] == "").sum()))

# 5. Class distribution by dataset
print("\n5. Class distribution:")
print(pd.crosstab(df["dataset"], df["canonical_label"]).to_string())

print("\n" + "=" * 65)
print("AUDIT FILES SAVED")
print("=" * 65)
print("audit_duplicate_image_ids.csv")
print("audit_exact_duplicate_sha_groups.csv")
print("audit_cross_dataset_sha_groups.csv")
print("audit_cross_split_group_sha_groups.csv")
print("audit_mixed_label_split_groups.csv")

print("\nAUDIT COMPLETE. Review findings before creating data splits.")

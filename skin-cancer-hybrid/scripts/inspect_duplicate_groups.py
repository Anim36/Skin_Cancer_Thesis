import pandas as pd
from pathlib import Path

base = Path("reports/dataset_inspection")

manifest = pd.read_csv(
    base / "master_dataset_manifest_sha_fixed.csv",
    dtype=str
)

duplicates = pd.read_csv(
    base / "audit_exact_duplicate_sha_groups.csv",
    dtype=str
)

cross = pd.read_csv(
    base / "audit_cross_split_group_sha_groups.csv",
    dtype=str
)

# Find all manifest rows belonging to duplicate SHA groups
all_details = manifest[
    manifest["sha256"].isin(duplicates["sha256"])
].copy()

cross_details = manifest[
    manifest["sha256"].isin(cross["sha256"])
].copy()

columns = [
    "dataset", "image_id", "image_path", "lesion_id",
    "canonical_label", "sha256", "split_group_id"
]

all_details[columns].sort_values(
    ["sha256", "dataset", "image_id"]
).to_csv(base / "audit_all_duplicate_image_details.csv", index=False)

cross_details[columns].sort_values(
    ["sha256", "dataset", "image_id"]
).to_csv(base / "audit_cross_split_duplicate_details.csv", index=False)

print("=" * 65)
print("CROSS-SPLIT DUPLICATE DETAILS")
print("=" * 65)

if cross_details.empty:
    print("No cross-split duplicate records.")
else:
    for sha, group in cross_details.groupby("sha256"):
        print("\nSHA256:", sha)
        print(group[[
            "dataset", "image_id", "lesion_id",
            "canonical_label", "split_group_id"
        ]].to_string(index=False))

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("All duplicate SHA groups:", len(duplicates))
print("Cross-split duplicate SHA groups:", len(cross))
print("Rows in all duplicate groups:", len(all_details))
print("Rows in cross-split groups:", len(cross_details))

print("\nSaved:")
print(base / "audit_all_duplicate_image_details.csv")
print(base / "audit_cross_split_duplicate_details.csv")

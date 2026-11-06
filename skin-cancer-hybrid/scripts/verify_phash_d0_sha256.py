from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "dataset_inspection"

inventory_file = REPORT / "image_sha256_inventory.csv"
pairs_file = REPORT / "phash_d0_isic2019_labels.csv"

inventory = pd.read_csv(inventory_file)
pairs = pd.read_csv(pairs_file)

# Build SHA256 lookup using the full image inventory.
sha_lookup = (
    inventory.groupby("relative_path")["sha256"]
    .first()
    .to_dict()
)

# Normalize paths for matching.
def normalize_path(value):
    return str(value).replace("/", "\\").lower()

inventory["path_key"] = inventory["relative_path"].apply(normalize_path)

path_to_sha = dict(zip(inventory["path_key"], inventory["sha256"]))

pairs["path_a"] = pairs["image_a"].apply(normalize_path)
pairs["path_b"] = pairs["image_b"].apply(normalize_path)

pairs["verified_sha256_a"] = pairs["path_a"].map(path_to_sha)
pairs["verified_sha256_b"] = pairs["path_b"].map(path_to_sha)

pairs["sha256_match"] = (
    pairs["verified_sha256_a"].notna()
    & pairs["verified_sha256_b"].notna()
    & (pairs["verified_sha256_a"] == pairs["verified_sha256_b"])
)

pairs["sha256_status"] = "NOT_VERIFIED"
pairs.loc[
    pairs["verified_sha256_a"].isna()
    | pairs["verified_sha256_b"].isna(),
    "sha256_status"
] = "PATH_NOT_FOUND"

pairs.loc[pairs["sha256_match"], "sha256_status"] = "EXACT_DUPLICATE"
pairs.loc[
    pairs["verified_sha256_a"].notna()
    & pairs["verified_sha256_b"].notna()
    & ~pairs["sha256_match"],
    "sha256_status"
] = "PHASH_ONLY_MATCH"

# Save detailed verification report.
output = REPORT / "phash_d0_sha256_verification.csv"
pairs.to_csv(output, index=False)

summary = (
    pairs["sha256_status"]
    .value_counts()
    .rename_axis("status")
    .reset_index(name="pair_count")
)
summary.to_csv(
    REPORT / "phash_d0_sha256_verification_summary.csv",
    index=False
)

print("\n" + "=" * 65)
print("PHASH DISTANCE-0 vs SHA256 VERIFICATION")
print("=" * 65)

print("Total pHash pairs:", len(pairs))
print("\nVerification summary:")
print(summary.to_string(index=False))

print("\nSHA256 match:", int(pairs["sha256_match"].sum()))
print("pHash-only match:", int((pairs["sha256_status"] == "PHASH_ONLY_MATCH").sum()))
print("Path not found:", int((pairs["sha256_status"] == "PATH_NOT_FOUND").sum()))

print("\nFirst 15 verified pairs:")
cols = [
    "image_id",
    "image_a",
    "image_b",
    "phash_distance",
    "isic2019_label",
    "sha256_status"
]
print(pairs[cols].head(15).to_string(index=False))

print("\nSaved:")
print(output)
print(REPORT / "phash_d0_sha256_verification_summary.csv")
print("\nVERIFICATION COMPLETE.")

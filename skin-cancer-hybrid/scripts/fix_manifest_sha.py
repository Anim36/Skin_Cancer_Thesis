import pandas as pd
from pathlib import Path

base = Path("reports/dataset_inspection")

sha_file = base / "image_sha256_inventory.csv"
manifest_file = base / "master_dataset_manifest.csv"

output_file = base / "master_dataset_manifest_sha_fixed.csv"

sha = pd.read_csv(sha_file, dtype=str)
manifest = pd.read_csv(manifest_file, dtype=str)

# Normalize image names and dataset names
sha["dataset_key"] = sha["dataset"].str.strip().str.upper()
sha["image_key"] = sha["image_name"].str.replace("\\", "/", regex=False).str.split("/").str[-1].str.strip().str.lower()

manifest["dataset_key"] = manifest["dataset"].str.strip().str.upper()
manifest["image_key"] = manifest["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1].str.strip().str.lower()

# Validate inventory uniqueness
duplicates = sha.duplicated(["dataset_key", "image_key"], keep=False)

if duplicates.any():
    print("ERROR: Duplicate dataset + image filename entries in SHA inventory:")
    print(sha.loc[duplicates, ["dataset", "image_name", "sha256"]].head(20).to_string(index=False))
    raise SystemExit("Fix duplicate SHA inventory entries before continuing.")

# Build lookup table
sha_lookup = sha[["dataset_key", "image_key", "sha256"]].copy()

# Remove old SHA values and merge correct values
manifest = manifest.drop(columns=["sha256"], errors="ignore")

fixed = manifest.merge(
    sha_lookup,
    on=["dataset_key", "image_key"],
    how="left",
    validate="one_to_one"
)

missing = fixed["sha256"].isna()
matched = len(fixed) - int(missing.sum())

print("\n========== SHA256 MAPPING RESULT ==========")
print("Manifest rows:", len(fixed))
print("SHA inventory rows:", len(sha))
print("Matched SHA256:", matched)
print("Missing SHA256:", int(missing.sum()))

if missing.any():
    print("\nMissing SHA examples:")
    print(
        fixed.loc[missing, ["dataset", "image_path", "image_key"]]
        .head(15)
        .to_string(index=False)
    )

# Cleanup temporary columns
fixed.drop(columns=["dataset_key", "image_key"], inplace=True)

# Save corrected copy
fixed.to_csv(output_file, index=False)

print("\nSaved:", output_file)

if missing.sum() == 0:
    print("\nSUCCESS: Every manifest image has a SHA256 hash.")
else:
    print("\nWARNING: Some images still have missing SHA256. Do not create splits yet.")

import pandas as pd
from pathlib import Path

base = Path("reports/dataset_inspection")

sha_path = base / "image_sha256_inventory.csv"
manifest_path = base / "master_dataset_manifest.csv"

sha = pd.read_csv(sha_path)
manifest = pd.read_csv(manifest_path)

print("\n========== SHA INVENTORY ==========")
print("Columns:", sha.columns.tolist())
print(sha.head(5).to_string(index=False))

print("\n========== MASTER MANIFEST ==========")
print("Columns:", manifest.columns.tolist())
print(manifest.head(5).to_string(index=False))

for col in ["relative_path", "image_path", "path"]:
    if col in sha.columns:
        print(f"\nSHA {col} samples:")
        print(sha[col].head(5).tolist())

    if col in manifest.columns:
        print(f"\nMANIFEST {col} samples:")
        print(manifest[col].head(5).tolist())

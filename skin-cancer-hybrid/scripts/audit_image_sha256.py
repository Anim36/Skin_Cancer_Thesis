import hashlib
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path("data/raw")
DATASETS = {
    "HAM10000": ROOT / "ham10000",
    "ISIC2018": ROOT / "isic2018",
    "ISIC2019": ROOT / "isic2019",
    "PAD_UFES20": ROOT / "pad_ufes20",
}
EXTENSIONS = {".jpg", ".jpeg", ".png"}

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

records = []
hash_groups = defaultdict(list)

print("Starting SHA256 image audit...")
print("This may take a few minutes.\n")

for dataset, folder in DATASETS.items():
    if not folder.exists():
        print(f"WARNING: Missing folder: {folder}")
        continue

    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    print(f"{dataset}: hashing {len(files)} images")

    for path in files:
        file_hash = sha256_file(path)
        record = {
            "dataset": dataset,
            "image_name": path.name,
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": file_hash,
            "file_size": path.stat().st_size,
        }
        records.append(record)
        hash_groups[file_hash].append(record)

df = pd.DataFrame(records)

out = Path("reports/dataset_inspection")
out.mkdir(parents=True, exist_ok=True)

# Save complete hash inventory
df.to_csv(out / "image_sha256_inventory.csv", index=False)

# Identify exact duplicate groups
duplicates = []

for file_hash, items in hash_groups.items():
    if len(items) > 1:
        datasets = sorted(set(x["dataset"] for x in items))
        duplicates.append({
            "sha256": file_hash,
            "total_copies": len(items),
            "dataset_count": len(datasets),
            "datasets": ";".join(datasets),
            "cross_dataset": len(datasets) > 1,
            "files": ";".join(
                f'{x["dataset"]}/{x["image_name"]}' for x in items
            ),
        })

dup_df = pd.DataFrame(duplicates)
dup_df.to_csv(out / "exact_duplicate_images_sha256.csv", index=False)

print("\n" + "=" * 55)
print("SHA256 AUDIT SUMMARY")
print("=" * 55)
print("Total image files:", len(df))
print("Unique SHA256 hashes:", df["sha256"].nunique())
print("Exact duplicate hash groups:", len(dup_df))

if len(dup_df):
    cross = dup_df[dup_df["cross_dataset"]]
    within = dup_df[~dup_df["cross_dataset"]]

    print("Cross-dataset duplicate groups:", len(cross))
    print("Within-dataset duplicate groups:", len(within))
    print("\nCross-dataset duplicate pairs by dataset:")
    for _, row in cross.head(20).iterrows():
        print(f'\n{row["datasets"]}: {row["total_copies"]} copies')
        print(row["files"])
else:
    print("No exact duplicate files found.")

print("\nImages by dataset:")
print(df.groupby("dataset").size().to_string())

print("\nSaved reports:")
print(out / "image_sha256_inventory.csv")
print(out / "exact_duplicate_images_sha256.csv")

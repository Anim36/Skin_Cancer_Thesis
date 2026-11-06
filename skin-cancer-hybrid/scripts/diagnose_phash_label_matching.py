from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "dataset_inspection"

pairs = pd.read_csv(REPORT / "phash_d0_same_image_id.csv")

def norm(x):
    if pd.isna(x):
        return ""
    return Path(str(x).replace("\\", "/")).stem.upper().replace(
        "_DOWNSAMPLED", ""
    )

pair_ids = set(pairs["normalized_id_a"].apply(norm))

files = {
    "HAM10000": (
        RAW / "ham10000" / "HAM10000_images" /
        "HAM10000_metadata"
    ),
    "ISIC2018": (
        RAW / "isic2018" /
        "ISIC2018_Task3_Training_GroundTruth" /
        "ISIC2018_Task3_Training_GroundTruth.csv"
    ),
    "ISIC2019": (
        RAW / "isic2019" /
        "ISIC_2019_Training_GroundTruth.csv"
    ),
}

print("\n" + "=" * 65)
print("IMAGE ID LABEL SOURCE MATCH DIAGNOSTIC")
print("=" * 65)

for name, path in files.items():
    print(f"\n{name}")
    print("File exists:", path.exists())

    if not path.exists():
        continue

    data = pd.read_csv(path)
    print("Columns:", data.columns.tolist())

    if "image" in data.columns:
        id_col = "image"
    elif "image_id" in data.columns:
        id_col = "image_id"
    else:
        print("No recognized image ID column.")
        continue

    source_ids = set(data[id_col].apply(norm))
    matched = pair_ids.intersection(source_ids)

    print("Source rows:", len(data))
    print("Unique source IDs:", len(source_ids))
    print("Matched pHash IDs:", len(matched))
    print("Unmatched pHash IDs:", len(pair_ids - source_ids))
    print("Sample matched IDs:", sorted(matched)[:10])

print("\nSample pHash IDs:")
print(sorted(pair_ids)[:20])

print("\nDIAGNOSTIC COMPLETE.")

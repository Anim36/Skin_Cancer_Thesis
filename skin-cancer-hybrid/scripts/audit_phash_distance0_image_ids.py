from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "dataset_inspection"

INPUT = REPORT / "disjoint_source_phash_candidates.csv"

df = pd.read_csv(INPUT)

def normalize_id(value):
    if pd.isna(value):
        return ""
    name = Path(str(value).replace("\\", "/")).stem
    if name.lower().endswith("_downsampled"):
        name = name[:-len("_downsampled")]
    return name.upper()

def image_dimensions(value):
    if pd.isna(value):
        return ""
    path = ROOT / "data" / "raw" / Path(str(value))
    if not path.exists():
        return "FILE_NOT_FOUND"
    try:
        with Image.open(path) as img:
            return f"{img.width}x{img.height}"
    except Exception as exc:
        return f"ERROR: {exc}"

# Only distance 0 candidates
d0 = df[df["phash_distance"] == 0].copy()

d0["normalized_id_a"] = d0["image_a"].apply(normalize_id)
d0["normalized_id_b"] = d0["image_b"].apply(normalize_id)

d0["same_image_id"] = (
    (d0["normalized_id_a"] != "") &
    (d0["normalized_id_a"] == d0["normalized_id_b"])
)

d0["dimensions_a"] = d0["image_a"].apply(image_dimensions)
d0["dimensions_b"] = d0["image_b"].apply(image_dimensions)

same_id = d0[d0["same_image_id"]].copy()
different_id = d0[~d0["same_image_id"]].copy()

same_path = REPORT / "phash_d0_same_image_id.csv"
different_path = REPORT / "phash_d0_different_image_id.csv"
full_path = REPORT / "phash_d0_id_dimension_audit.csv"

same_id.to_csv(same_path, index=False)
different_id.to_csv(different_path, index=False)
d0.to_csv(full_path, index=False)

print("\n" + "=" * 60)
print("DISTANCE-0 IMAGE ID AND DIMENSION AUDIT")
print("=" * 60)

print(f"Total Distance-0 pairs: {len(d0)}")
print(f"Same normalized Image ID: {len(same_id)}")
print(f"Different normalized Image ID: {len(different_id)}")

print("\nImage dimension combinations:")
print(
    same_id.groupby(
        ["dimensions_a", "dimensions_b"],
        dropna=False
    ).size().sort_values(ascending=False).head(20).to_string()
)

print("\nFIRST 20 SAME-ID PAIRS:")
columns = [
    "image_a", "image_b",
    "normalized_id_a",
    "dimensions_a", "dimensions_b",
    "sha256_a", "sha256_b"
]
print(same_id[columns].head(20).to_string(index=False))

print("\nSaved reports:")
print(same_path)
print(different_path)
print(full_path)

print("\nAUDIT COMPLETE.")

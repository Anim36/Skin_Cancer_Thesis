from pathlib import Path
import pandas as pd
import hashlib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "dataset_inspection"
REPORT.mkdir(parents=True, exist_ok=True)

HAM_META = RAW / "ham10000" / "HAM10000_images" / "HAM10000_metadata"
ISIC19_GT = RAW / "isic2019" / "ISIC_2019_Training_GroundTruth.csv"
ISIC19_META = RAW / "isic2019" / "ISIC_2019_Training_Metadata.csv"
SHA_FILE = REPORT / "image_sha256_inventory.csv"

TARGET_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
ISIC19_ALLOWED = {"MEL", "NV", "BCC", "BKL", "DF", "VASC"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def normalize_id(value):
    return str(value).strip().upper()

def normalize_path(value):
    return str(value).replace("/", "\\").lower()

def index_images(folder):
    result = {}
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_id = normalize_id(path.stem)
            if image_id.endswith("_DOWNSAMPLED"):
                image_id = image_id[:-12]
            result.setdefault(image_id, []).append(path)

    return result

def select_image(paths):
    # Prefer the non-downsampled/original file when multiple files share an ID.
    return sorted(
        paths,
        key=lambda p: ("downsampled" in p.stem.lower(), len(str(p)))
    )[0]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def make_row(path, dataset, image_id, native_label, canonical_label,
             lesion_id=None, patient_id=None):
    relative = path.relative_to(ROOT).as_posix()
    return {
        "image_path": relative,
        "dataset": dataset,
        "image_id": image_id,
        "lesion_id": lesion_id,
        "patient_id": patient_id,
        "native_label": native_label,
        "canonical_label": canonical_label,
        "sha256": None,
        "phash_group": f"IMAGE:{image_id}",
        "split_group_id": (
            f"{dataset}:LESION:{lesion_id}"
            if pd.notna(lesion_id) and str(lesion_id).strip()
            else None
        ),
        "split_eligible": bool(
            pd.notna(lesion_id) and str(lesion_id).strip()
        ),
        "label_status": "VALIDATED_7_CLASS"
    }

print("Indexing HAM10000 images...")
ham_folder = RAW / "ham10000" / "HAM10000_images"
ham_images = index_images(ham_folder)

print("Indexing ISIC2019 images...")
isic19_folder = RAW / "isic2019"
isic19_images = index_images(isic19_folder)

print("Reading metadata...")
ham = pd.read_csv(HAM_META)
gt19 = pd.read_csv(ISIC19_GT)
meta19 = pd.read_csv(ISIC19_META)

# ---------------- HAM10000 ----------------
ham_rows = []
ham_missing_images = []

for _, r in ham.iterrows():
    image_id = normalize_id(r["image_id"])
    label = str(r["dx"]).strip().upper()

    if label not in TARGET_CLASSES:
        continue

    paths = ham_images.get(image_id, [])
    if not paths:
        ham_missing_images.append(image_id)
        continue

    path = select_image(paths)
    ham_rows.append(
        make_row(
            path, "HAM10000", image_id, label, label,
            lesion_id=r["lesion_id"]
        )
    )

ham_ids = set(ham["image_id"].map(normalize_id))

# ---------------- ISIC2019 ----------------
gt19["image_id"] = gt19["image"].map(normalize_id)

for col in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]:
    gt19[col] = pd.to_numeric(gt19[col], errors="coerce").fillna(0)

gt19["positive_count"] = gt19[
    ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
].eq(1).sum(axis=1)

gt19["native_label"] = gt19.apply(
    lambda r: (
        next(
            c for c in
            ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
            if r[c] == 1
        )
        if r["positive_count"] == 1 else None
    ),
    axis=1
)

meta19["image_id"] = meta19["image"].map(normalize_id)
meta19_small = meta19[["image_id", "lesion_id"]].drop_duplicates("image_id")

isic19 = gt19.merge(meta19_small, on="image_id", how="left")

isic19_rows = []
exclusions = []

for _, r in isic19.iterrows():
    image_id = r["image_id"]
    label = r["native_label"]

    reason = None

    if r["positive_count"] != 1:
        reason = "INVALID_OR_MULTI_LABEL"
    elif label not in ISIC19_ALLOWED:
        reason = f"NOT_IN_7_CLASS_POLICY:{label}"
    elif image_id in ham_ids:
        reason = "IMAGE_ID_OVERLAPS_HAM10000"
    elif pd.isna(r["lesion_id"]) or not str(r["lesion_id"]).strip():
        reason = "MISSING_LESION_ID"

    if reason:
        exclusions.append({
            "dataset": "ISIC2019",
            "image_id": image_id,
            "native_label": label,
            "reason": reason
        })
        continue

    paths = isic19_images.get(image_id, [])
    if not paths:
        exclusions.append({
            "dataset": "ISIC2019",
            "image_id": image_id,
            "native_label": label,
            "reason": "IMAGE_FILE_NOT_FOUND"
        })
        continue

    path = select_image(paths)
    isic19_rows.append(
        make_row(
            path, "ISIC2019", image_id, label, label,
            lesion_id=r["lesion_id"]
        )
    )

# ---------------- Build manifest ----------------
manifest = pd.DataFrame(ham_rows + isic19_rows)
exclusion_df = pd.DataFrame(exclusions)

# Attach SHA256 values from the existing inventory.
if SHA_FILE.exists():
    sha_df = pd.read_csv(SHA_FILE)
    sha_df["path_key"] = sha_df["relative_path"].apply(normalize_path)
    sha_map = dict(zip(sha_df["path_key"], sha_df["sha256"]))

    manifest["path_key"] = manifest["image_path"].apply(normalize_path)
    manifest["sha256"] = manifest["path_key"].map(sha_map)
    manifest.drop(columns=["path_key"], inplace=True)

manifest_file = REPORT / "master_dataset_manifest.csv"
exclusion_file = REPORT / "master_dataset_exclusions.csv"

manifest.to_csv(manifest_file, index=False)
exclusion_df.to_csv(exclusion_file, index=False)

# ---------------- Summary ----------------
print("\n" + "=" * 65)
print("MASTER DATASET MANIFEST SUMMARY")
print("=" * 65)

print("\nIncluded samples by dataset:")
print(manifest.groupby("dataset").size().to_string())

print("\nCanonical class distribution:")
print(
    manifest.groupby(["dataset", "canonical_label"])
    .size().unstack(fill_value=0).to_string()
)

print("\nTotal included:", len(manifest))
print("Total exclusions:", len(exclusion_df))
print("Missing HAM images:", len(ham_missing_images))
print("Missing SHA256:", int(manifest["sha256"].isna().sum()))
print("Missing split group:", int(manifest["split_group_id"].isna().sum()))

print("\nISIC2019 exclusion reasons:")
if len(exclusion_df):
    print(exclusion_df["reason"].value_counts().to_string())

print("\nUnique image IDs:", manifest["image_id"].nunique())
print("Unique split groups:", manifest["split_group_id"].nunique())

print("\nSaved:")
print(manifest_file)
print(exclusion_file)

print("\nMANIFEST BUILD COMPLETE.")

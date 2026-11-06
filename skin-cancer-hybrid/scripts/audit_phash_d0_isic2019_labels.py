from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "dataset_inspection"

PAIR_FILE = REPORT / "phash_d0_same_image_id.csv"
GT_FILE = (
    ROOT / "data" / "raw" / "isic2019" /
    "ISIC_2019_Training_GroundTruth.csv"
)

pairs = pd.read_csv(PAIR_FILE)
gt = pd.read_csv(GT_FILE)

classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

def normalize_id(value):
    if pd.isna(value):
        return ""
    return Path(str(value).replace("\\", "/")).stem.upper().replace(
        "_DOWNSAMPLED", ""
    )

pairs["image_id"] = pairs["normalized_id_a"].apply(normalize_id)
gt["image_id"] = gt["image"].apply(normalize_id)

# Validate one-hot / multi-hot ground-truth rows.
for col in classes:
    gt[col] = pd.to_numeric(gt[col], errors="coerce")

gt["positive_class_count"] = gt[classes].eq(1).sum(axis=1)

gt["isic2019_label"] = gt.apply(
    lambda row: (
        [c for c in classes if row[c] == 1][0]
        if row["positive_class_count"] == 1
        else (
            "NO_POSITIVE_LABEL"
            if row["positive_class_count"] == 0
            else "MULTI_LABEL"
        )
    ),
    axis=1
)

gt_small = gt[
    ["image_id", "isic2019_label", "positive_class_count"] + classes
].drop_duplicates("image_id")

result = pairs.merge(
    gt_small,
    on="image_id",
    how="left",
    indicator=True
)

result["label_match_status"] = result["_merge"].map({
    "both": "LABEL_FOUND",
    "left_only": "LABEL_NOT_FOUND"
})
result.drop(columns=["_merge"], inplace=True)

found = result[result["label_match_status"] == "LABEL_FOUND"].copy()
missing = result[result["label_match_status"] == "LABEL_NOT_FOUND"].copy()

# Save full report and separate label categories.
found.to_csv(REPORT / "phash_d0_isic2019_labels.csv", index=False)
missing.to_csv(REPORT / "phash_d0_isic2019_labels_missing.csv", index=False)

found.groupby("isic2019_label").size().rename(
    "pair_count"
).reset_index().to_csv(
    REPORT / "phash_d0_isic2019_label_summary.csv", index=False
)

print("\n" + "=" * 65)
print("ISIC 2019 GROUND TRUTH AUDIT FOR DISTANCE-0 PAIRS")
print("=" * 65)

print("Total pairs:", len(result))
print("Labels found:", len(found))
print("Labels missing:", len(missing))

print("\nISIC 2019 label distribution:")
print(found["isic2019_label"].value_counts(dropna=False).to_string())

print("\nPositive class count distribution:")
print(found["positive_class_count"].value_counts(dropna=False).sort_index().to_string())

print("\nFIRST 20 MATCHED PAIRS:")
columns = [
    "image_id", "image_a", "image_b",
    "phash_distance", "isic2019_label",
    "positive_class_count"
]
print(found[columns].head(20).to_string(index=False))

print("\nMULTI-LABEL OR INVALID ROWS:")
invalid = found[found["positive_class_count"] != 1]
print("Count:", len(invalid))
if len(invalid):
    print(invalid[columns].head(20).to_string(index=False))

print("\nSAVED:")
print(REPORT / "phash_d0_isic2019_labels.csv")
print(REPORT / "phash_d0_isic2019_labels_missing.csv")
print(REPORT / "phash_d0_isic2019_label_summary.csv")

print("\nAUDIT COMPLETE.")

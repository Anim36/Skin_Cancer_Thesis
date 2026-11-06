from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "dataset_inspection"

PHASH_FILE = REPORT / "phash_d0_same_image_id.csv"
LABEL_FILE = REPORT / "cross_dataset_label_comparison.csv"

for file in [PHASH_FILE, LABEL_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")

pairs = pd.read_csv(PHASH_FILE)
labels = pd.read_csv(LABEL_FILE)

def normalize_id(value):
    if pd.isna(value):
        return ""
    return Path(str(value).replace("\\", "/")).stem.upper().replace(
        "_DOWNSAMPLED", ""
    )

# Normalize IDs for a safe join.
pairs["join_image_id"] = pairs["normalized_id_a"].apply(normalize_id)
labels["join_image_id"] = labels["image_id"].apply(normalize_id)

# Ensure one label record per image ID.
duplicates = labels[
    labels["join_image_id"].duplicated(keep=False)
]

if len(duplicates):
    print("WARNING: Duplicate image IDs in label report:", len(duplicates))
    print(duplicates[["image_id", "dx", "label_18", "label_19"]].head(10))

labels = labels.drop_duplicates("join_image_id", keep="first")

label_columns = [
    "join_image_id", "lesion_id", "dx", "label_18", "label_19"
]

result = pairs.merge(
    labels[label_columns],
    on="join_image_id",
    how="left",
    indicator=True
)

def clean_label(value):
    if pd.isna(value):
        return None
    return str(value).strip().upper()

for col in ["dx", "label_18", "label_19"]:
    result[col + "_normalized"] = result[col].apply(clean_label)

def compare_labels(row):
    a = row["label_18_normalized"]
    b = row["label_19_normalized"]

    if a is None or b is None:
        return "LABEL_NOT_AVAILABLE"

    if a == b:
        return "SAME_LABEL"

    return "LABEL_CONFLICT"

result["isic18_vs_isic19_status"] = result.apply(
    compare_labels, axis=1
)

def ham_vs_isic18(row):
    a = row["dx_normalized"]
    b = row["label_18_normalized"]

    if a is None or b is None:
        return "LABEL_NOT_AVAILABLE"

    return "SAME_LABEL" if a == b else "LABEL_CONFLICT"

result["ham_vs_isic18_status"] = result.apply(
    ham_vs_isic18, axis=1
)

def ham_vs_isic19(row):
    a = row["dx_normalized"]
    b = row["label_19_normalized"]

    if a is None or b is None:
        return "LABEL_NOT_AVAILABLE"

    return "SAME_LABEL" if a == b else "LABEL_CONFLICT"

result["ham_vs_isic19_status"] = result.apply(
    ham_vs_isic19, axis=1
)

result["label_record_found"] = result["_merge"].eq("both")
result.drop(columns=["_merge"], inplace=True)

# Save complete joined audit.
full_path = REPORT / "phash_d0_label_comparison_full.csv"
result.to_csv(full_path, index=False)

# Save separate reports.
conflicts = result[
    result["isic18_vs_isic19_status"] == "LABEL_CONFLICT"
].copy()

same = result[
    result["isic18_vs_isic19_status"] == "SAME_LABEL"
].copy()

unavailable = result[
    result["isic18_vs_isic19_status"] == "LABEL_NOT_AVAILABLE"
].copy()

conflicts.to_csv(
    REPORT / "phash_d0_label_conflicts.csv", index=False
)

same.to_csv(
    REPORT / "phash_d0_same_labels.csv", index=False
)

unavailable.to_csv(
    REPORT / "phash_d0_labels_unavailable.csv", index=False
)

print("\n" + "=" * 65)
print("PHASH DISTANCE-0 LABEL COMPARISON AUDIT")
print("=" * 65)

print("Total same-ID pHash pairs:", len(result))
print("Pairs with label record:", int(result["label_record_found"].sum()))
print("Pairs without label record:", len(unavailable))

print("\nISIC 2018 vs ISIC 2019:")
print(result["isic18_vs_isic19_status"].value_counts().to_string())

print("\nHAM10000 vs ISIC 2018:")
print(result["ham_vs_isic18_status"].value_counts().to_string())

print("\nHAM10000 vs ISIC 2019:")
print(result["ham_vs_isic19_status"].value_counts().to_string())

print("\nISIC 2018 vs ISIC 2019 CONFLICT LABEL COMBINATIONS:")
if len(conflicts):
    print(
        conflicts.groupby(
            ["label_18_normalized", "label_19_normalized"],
            dropna=False
        ).size().sort_values(ascending=False).to_string()
    )
else:
    print("No conflicts found among available labels.")

print("\nFIRST 20 CONFLICTS:")
columns = [
    "join_image_id",
    "image_a",
    "image_b",
    "label_18",
    "label_19",
    "dx",
    "lesion_id"
]
print(conflicts[columns].head(20).to_string(index=False))

print("\nSAVED REPORTS:")
print(full_path)
print(REPORT / "phash_d0_label_conflicts.csv")
print(REPORT / "phash_d0_same_labels.csv")
print(REPORT / "phash_d0_labels_unavailable.csv")

print("\nAUDIT COMPLETE.")

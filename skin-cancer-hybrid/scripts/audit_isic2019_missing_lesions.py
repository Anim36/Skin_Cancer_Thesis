import pandas as pd
from pathlib import Path

# Load metadata
i19_file = next(Path("data/raw/isic2019").rglob("ISIC_2019_Training_Metadata.csv"))
i19 = pd.read_csv(i19_file)

# Load ground truth
gt_file = next(Path("data/raw/isic2019").rglob("ISIC_2019_Training_GroundTruth.csv"))
gt = pd.read_csv(gt_file)

# Load HAM10000 metadata
ham_file = next(Path("data/raw/ham10000").rglob("HAM10000_metadata"))
ham = pd.read_csv(ham_file)

# Join ISIC 2019 metadata with labels
merged = i19.merge(gt, on="image", how="left")

missing = merged[merged["lesion_id"].isna()].copy()
known = merged[merged["lesion_id"].notna()].copy()

print("ISIC 2019 missing lesion ID analysis")
print("-" * 45)
print("Total images:", len(merged))
print("Missing lesion IDs:", len(missing))
print("Known lesion IDs:", len(known))

print("\nMissing lesion IDs by class:")
class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
for col in class_cols:
    if col in missing.columns:
        print(f"{col}: {int(missing[col].sum())}")

print("\nMissing lesion rows with multiple positive labels:",
      int((missing[class_cols].sum(axis=1) > 1).sum()))

print("\nHAM10000 overlap:")
ham_ids = set(ham["image_id"].astype(str))
missing["image_id"] = missing["image"].astype(str)
overlap = missing[missing["image_id"].isin(ham_ids)]
print("Missing-ID images also in HAM10000:", len(overlap))

print("\nKnown lesion grouping:")
print("Unique known lesions:", known["lesion_id"].nunique())
print("Images in multi-image lesions:",
      int(known.groupby("lesion_id")["image"].transform("size").gt(1).sum()))
print("Maximum images per lesion:",
      int(known.groupby("lesion_id").size().max()))

print("\nSample missing-ID records:")
print(missing[["image"] + class_cols].head(10).to_string(index=False))

# Save detailed report
out = Path("reports/dataset_inspection")
out.mkdir(parents=True, exist_ok=True)
missing.to_csv(out / "isic2019_missing_lesion_ids.csv", index=False)
print("\nSaved: reports/dataset_inspection/isic2019_missing_lesion_ids.csv")

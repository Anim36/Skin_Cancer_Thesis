import pandas as pd
from pathlib import Path

ham_root = Path("data/raw/ham10000")
isic18_root = Path("data/raw/isic2018")
isic19_root = Path("data/raw/isic2019")

ham_file = next(ham_root.rglob("HAM10000_metadata"))
isic18_file = next(isic18_root.rglob("ISIC2018_Task3_Training_GroundTruth.csv"))
isic19_file = next(isic19_root.rglob("ISIC_2019_Training_GroundTruth.csv"))

ham = pd.read_csv(ham_file)
i18 = pd.read_csv(isic18_file)
i19 = pd.read_csv(isic19_file)

# Convert one-hot labels into class names
i18["label_18"] = i18[
    ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
].idxmax(axis=1)

i19_classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
i19["label_19"] = i19[i19_classes].idxmax(axis=1)

# Join by exact image ID
merged = ham[["image_id", "lesion_id", "dx"]].merge(
    i18[["image", "label_18"]],
    left_on="image_id",
    right_on="image",
    how="inner"
).merge(
    i19[["image", "label_19"]],
    left_on="image_id",
    right_on="image",
    how="inner"
)

print("Common images:", len(merged))
print("\nHAM10000 vs ISIC 2018:")
print(pd.crosstab(merged["dx"], merged["label_18"]))

print("\nHAM10000 vs ISIC 2019:")
print(pd.crosstab(merged["dx"], merged["label_19"]))

print("\nISIC 2018 vs ISIC 2019:")
print(pd.crosstab(merged["label_18"], merged["label_19"]))

# Save detailed comparison
out = Path("reports/dataset_inspection")
out.mkdir(parents=True, exist_ok=True)
merged.to_csv(out / "cross_dataset_label_comparison.csv", index=False)

print("\nSaved: reports/dataset_inspection/cross_dataset_label_comparison.csv")

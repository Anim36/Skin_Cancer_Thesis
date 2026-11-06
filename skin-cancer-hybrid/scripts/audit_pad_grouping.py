import pandas as pd
from pathlib import Path

file = next(Path("data/raw/pad_ufes20").rglob("metadata.csv"))
df = pd.read_csv(file)

print("PAD-UFES-20 GROUPING AUDIT")
print("=" * 50)
print("Total images:", len(df))
print("Unique image IDs:", df["img_id"].nunique())

for col in ["patient_id", "lesion_id"]:
    print(f"\n{col}")
    print("Missing:", df[col].isna().sum())
    print("Unique non-missing:", df[col].nunique())

    valid = df[df[col].notna()]
    counts = valid.groupby(col).size()

    print("Groups with multiple images:", int((counts > 1).sum()))
    print("Maximum images per group:", int(counts.max()))
    print("Group size distribution:")
    print(counts.value_counts().sort_index().to_string())

print("\nPatient-to-lesion relationship")
valid = df.dropna(subset=["patient_id", "lesion_id"])
lesions_per_patient = valid.groupby("patient_id")["lesion_id"].nunique()

print("Patients with multiple lesions:",
      int((lesions_per_patient > 1).sum()))
print("Maximum lesions per patient:",
      int(lesions_per_patient.max()))

print("\nImages per diagnostic class:")
print(df["diagnostic"].value_counts(dropna=False).to_string())

print("\nPatients with missing patient_id:",
      df.loc[df["patient_id"].isna(), "img_id"].nunique())

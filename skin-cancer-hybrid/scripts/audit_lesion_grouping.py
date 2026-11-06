import pandas as pd
from pathlib import Path

# HAM10000
ham_file = next(Path("data/raw/ham10000").rglob("HAM10000_metadata"))
ham = pd.read_csv(ham_file)

# ISIC 2018
i18_root = Path("data/raw/isic2018")
i18_file = next(i18_root.rglob("ISIC2018_Task3_Training_LesionGroupings.csv"))
i18 = pd.read_csv(i18_file)

# ISIC 2019
i19_file = next(Path("data/raw/isic2019").rglob("ISIC_2019_Training_Metadata.csv"))
i19 = pd.read_csv(i19_file)

for name, df in [
    ("HAM10000", ham),
    ("ISIC2018", i18),
    ("ISIC2019", i19),
]:
    print(f"\n{name}")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "lesion_id" in df.columns:
        print("Unique lesions:", df["lesion_id"].nunique())
        print("Missing lesion IDs:", df["lesion_id"].isna().sum())
        print("Lesions with multiple images:",
              (df.groupby("lesion_id").size() > 1).sum())
    else:
        print("No lesion_id column")

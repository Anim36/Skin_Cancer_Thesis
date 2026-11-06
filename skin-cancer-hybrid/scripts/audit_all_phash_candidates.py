from pathlib import Path
import pandas as pd
import re

root = Path(__file__).resolve().parents[1]
report = root / "reports" / "dataset_inspection"

source = report / "perceptual_hash_candidates_with_provenance.csv"
df = pd.read_csv(source)

def get_image_id(path):
    name = Path(str(path).replace("\\", "/")).stem.upper()
    name = re.sub(r"_DOWNSAMPLED$", "", name)
    match = re.search(r"ISIC_\d+", name)
    return match.group(0) if match else None

df["image_id_a"] = df["image_a"].apply(get_image_id)
df["image_id_b"] = df["image_b"].apply(get_image_id)

df["same_image_id"] = (
    df["image_id_a"].notna()
    & df["image_id_b"].notna()
    & (df["image_id_a"] == df["image_id_b"])
)

df["same_sha256"] = (
    df["sha256_a"].notna()
    & df["sha256_b"].notna()
    & (df["sha256_a"] == df["sha256_b"])
)

# Summaries
print("\n" + "=" * 65)
print("FULL PERCEPTUAL HASH CANDIDATE AUDIT")
print("=" * 65)

print("\nTotal candidates:", len(df))

print("\nDistance distribution:")
print(df["phash_distance"].value_counts().sort_index().to_string())

print("\nSame normalized image ID:")
print(df["same_image_id"].value_counts().to_string())

print("\nSame SHA256:")
print(df["same_sha256"].value_counts().to_string())

print("\nSame image ID by pHash distance:")
print(pd.crosstab(
    df["phash_distance"],
    df["same_image_id"]
).to_string())

print("\nDisjoint source sets:")
print(df["disjoint_source_sets"].value_counts(dropna=False).to_string())

# Export same-ID candidates.
same_id = df[df["same_image_id"]].copy()
same_id.to_csv(report / "phash_same_image_id_candidates.csv", index=False)

# Export disjoint-source candidates with same ID.
same_id_disjoint = same_id[
    same_id["disjoint_source_sets"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
]
same_id_disjoint.to_csv(
    report / "phash_same_id_disjoint_source_candidates.csv",
    index=False
)

# Export summary by distance.
summary = (
    df.groupby("phash_distance")
    .agg(
        total_pairs=("phash_distance", "size"),
        same_image_id=("same_image_id", "sum"),
        exact_sha256=("same_sha256", "sum")
    )
    .reset_index()
)
summary.to_csv(report / "phash_full_candidate_summary.csv", index=False)

print("\nSame-ID candidate count:", len(same_id))
print("Same-ID + disjoint-source count:", len(same_id_disjoint))

print("\nFirst 20 same-ID candidates:")
cols = [
    "image_id_a", "image_a", "image_b",
    "phash_distance", "same_sha256",
    "datasets_a", "datasets_b"
]
print(same_id[cols].head(20).to_string(index=False))

print("\nSaved reports:")
print(report / "phash_same_image_id_candidates.csv")
print(report / "phash_same_id_disjoint_source_candidates.csv")
print(report / "phash_full_candidate_summary.csv")

print("\nAUDIT COMPLETE.")

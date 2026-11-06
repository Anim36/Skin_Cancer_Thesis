from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
report = root / "reports" / "dataset_inspection"

df = pd.read_csv(report / "phash_same_image_id_candidates.csv")

print("\n" + "=" * 65)
print("SAME IMAGE ID - SOURCE DATASET DISTRIBUTION")
print("=" * 65)

print("\nDataset pair distribution:")
print(
    df.groupby(["datasets_a", "datasets_b"], dropna=False)
      .size()
      .sort_values(ascending=False)
      .to_string()
)

print("\nDistance distribution:")
print(
    df.groupby("phash_distance")
      .size()
      .to_string()
)

print("\nUnique normalized Image IDs:", df["image_id_a"].nunique())

print("\nImage IDs appearing more than once:")
counts = df["image_id_a"].value_counts()
repeated = counts[counts > 1]
print("Repeated IDs:", len(repeated))
print(repeated.head(30).to_string())

# Export distinct IDs with their pair counts.
repeated.rename_axis("image_id").reset_index(name="pair_count").to_csv(
    report / "repeated_phash_image_ids.csv", index=False
)

df.groupby(["datasets_a", "datasets_b", "phash_distance"]).size().reset_index(
    name="pair_count"
).to_csv(
    report / "same_image_id_dataset_distance_summary.csv", index=False
)

print("\nReports saved:")
print(report / "repeated_phash_image_ids.csv")
print(report / "same_image_id_dataset_distance_summary.csv")

print("\nAUDIT COMPLETE.")

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "dataset_inspection"

INPUT = REPORT_DIR / "perceptual_hash_candidates_with_provenance.csv"

if not INPUT.exists():
    raise FileNotFoundError(f"Input report not found: {INPUT}")

df = pd.read_csv(INPUT)

print("\nINPUT COLUMNS:")
print(df.columns.tolist())

required = {"sources_a", "sources_b"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

def parse_sources(value):
    if pd.isna(value):
        return set()
    return {x.strip() for x in str(value).split(";") if x.strip()}

# Keep pairs whose source dataset memberships do not overlap.
source_a = df["sources_a"].apply(parse_sources)
source_b = df["sources_b"].apply(parse_sources)

disjoint_mask = [
    len(a) > 0 and len(b) > 0 and a.isdisjoint(b)
    for a, b in zip(source_a, source_b)
]

disjoint = df.loc[disjoint_mask].copy()

distance_col = "phash_distance"
if distance_col not in disjoint.columns:
    raise ValueError(
        f"'{distance_col}' column not found. "
        f"Available columns: {disjoint.columns.tolist()}"
    )

disjoint[distance_col] = pd.to_numeric(
    disjoint[distance_col], errors="coerce"
)

disjoint = disjoint.sort_values(
    [distance_col], ascending=True
)

output = REPORT_DIR / "disjoint_source_phash_candidates.csv"
disjoint.to_csv(output, index=False)

print("\n" + "=" * 60)
print("DISJOINT-SOURCE PHASH REVIEW REPORT")
print("=" * 60)
print(f"Total original candidate pairs: {len(df)}")
print(f"Disjoint-source candidate pairs: {len(disjoint)}")
print("\nHamming distance distribution:")
print(disjoint[distance_col].value_counts().sort_index().to_string())

# Export separate review files for each distance.
for distance in sorted(disjoint[distance_col].dropna().unique()):
    subset = disjoint[disjoint[distance_col] == distance]
    filename = f"disjoint_phash_distance_{int(distance)}.csv"
    path = REPORT_DIR / filename
    subset.to_csv(path, index=False)
    print(f"Saved {filename}: {len(subset)} pairs")

# Prioritize the closest visual matches.
priority = disjoint[disjoint[distance_col] <= 2]
priority_path = REPORT_DIR / "disjoint_phash_priority_distance_0_2.csv"
priority.to_csv(priority_path, index=False)

print(f"\nPriority pairs (distance 0 or 2): {len(priority)}")
print(f"\nFull report saved:\n{output}")

# Show a compact preview without hiding the original CSV.
preview_cols = [
    c for c in [
        "phash_distance",
        "sources_a",
        "sources_b",
        "shared_source_datasets",
        "disjoint_source_sets",
        "sha256_a",
        "sha256_b",
        "path_a",
        "path_b",
        "image_a",
        "image_b",
    ] if c in disjoint.columns
]

print("\nFIRST 15 PRIORITY PAIRS:")
if len(priority):
    print(priority[preview_cols].head(15).to_string(index=False))
else:
    print("No pairs with distance 0 or 2.")

print("\nAUDIT COMPLETE.")

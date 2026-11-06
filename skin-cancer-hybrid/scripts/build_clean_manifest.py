import pandas as pd
from pathlib import Path

base = Path("reports/dataset_inspection")
source = base / "master_dataset_manifest_sha_fixed.csv"

df = pd.read_csv(source, dtype=str).fillna("")

required = [
    "dataset", "image_id", "canonical_label",
    "sha256", "split_group_id"
]
for col in required:
    if col not in df.columns:
        raise SystemExit(f"Missing column: {col}")

for col in required:
    df[col] = df[col].str.strip()

if (df["sha256"].str.len() != 64).any():
    raise SystemExit("Missing or invalid SHA256. Stop.")

# Union-Find: connect lesion groups sharing exact image hashes
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for sha, group in df.groupby("sha256"):
    groups = sorted(group["split_group_id"].unique())
    if len(groups) > 1:
        for other in groups[1:]:
            union(groups[0], other)

# Assign connected component IDs
df["component_root"] = df["split_group_id"].map(find)

component_stats = df.groupby("component_root").agg(
    group_count=("split_group_id", "nunique"),
    label_count=("canonical_label", "nunique"),
    labels=("canonical_label", lambda x: "|".join(sorted(set(x))))
).reset_index()

connected = component_stats[component_stats["group_count"] > 1]
conflict_roots = set(
    connected.loc[connected["label_count"] > 1, "component_root"]
)

# Quarantine every record in conflicting connected components
quarantine = df[df["component_root"].isin(conflict_roots)].copy()
remaining = df[~df["component_root"].isin(conflict_roots)].copy()

# Merge same-label connected lesion groups into one split group
connected_roots = set(connected["component_root"]) - conflict_roots

remaining["original_split_group_id"] = remaining["split_group_id"]

mask = remaining["component_root"].isin(connected_roots)
remaining.loc[mask, "split_group_id"] = (
    "DUPCOMP:" + remaining.loc[mask, "component_root"]
)

# Deduplicate every exact SHA256: keep deterministic representative
remaining = remaining.sort_values(
    ["dataset", "image_id", "image_path"]
).copy()

duplicate_mask = remaining.duplicated("sha256", keep="first")

removed = remaining[duplicate_mask].copy()
clean = remaining[~duplicate_mask].copy()

# Export
quarantine.to_csv(
    base / "quarantined_conflicting_duplicates.csv", index=False
)
removed.to_csv(
    base / "removed_exact_duplicate_copies.csv", index=False
)
clean.drop(columns=["component_root", "original_split_group_id"],
           errors="ignore").to_csv(
    base / "master_dataset_manifest_clean.csv", index=False
)

# Final integrity checks
remaining_sha_duplicates = int(clean.duplicated("sha256").sum())

mixed_groups = (
    clean.groupby("split_group_id")["canonical_label"].nunique() > 1
).sum()

print("=" * 65)
print("CLEAN MANIFEST BUILD SUMMARY")
print("=" * 65)
print("Original manifest rows:", len(df))
print("Conflicting component roots:", len(conflict_roots))
print("Quarantined records:", len(quarantine))
print("Removed exact duplicate copies:", len(removed))
print("Clean manifest rows:", len(clean))
print("Unique SHA256 in clean manifest:", clean["sha256"].nunique())
print("Remaining duplicate SHA rows:", remaining_sha_duplicates)
print("Mixed-label split groups:", int(mixed_groups))
print("Unique split groups:", clean["split_group_id"].nunique())

print("\nClass distribution:")
print(clean.groupby(["dataset", "canonical_label"]).size().unstack(fill_value=0).to_string())

print("\nSaved:")
print(base / "master_dataset_manifest_clean.csv")
print(base / "quarantined_conflicting_duplicates.csv")
print(base / "removed_exact_duplicate_copies.csv")

if remaining_sha_duplicates == 0 and mixed_groups == 0:
    print("\nSUCCESS: Clean manifest integrity checks passed.")
else:
    print("\nWARNING: Integrity checks failed. Do not create splits.")

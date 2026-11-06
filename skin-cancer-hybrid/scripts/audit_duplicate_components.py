import pandas as pd
from pathlib import Path

base = Path("reports/dataset_inspection")
df = pd.read_csv(
    base / "master_dataset_manifest_sha_fixed.csv",
    dtype=str
).fillna("")

for col in ["sha256", "split_group_id", "canonical_label"]:
    df[col] = df[col].str.strip()

df = df[df["sha256"].str.len() == 64].copy()

# Union-Find over split groups connected by identical SHA256
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

duplicate_hashes = (
    df.groupby("sha256")["split_group_id"]
      .agg(lambda x: sorted(set(x)))
)

duplicate_hashes = duplicate_hashes[
    duplicate_hashes.apply(len) > 1
]

for groups in duplicate_hashes:
    first = groups[0]
    for other in groups[1:]:
        union(first, other)

# Build connected components
df["component_id"] = df["split_group_id"].map(find)

component_summary = df.groupby("component_id").agg(
    image_count=("image_id", "size"),
    lesion_group_count=("split_group_id", "nunique"),
    sha_count=("sha256", "nunique"),
    label_count=("canonical_label", "nunique"),
    labels=("canonical_label", lambda x: "|".join(sorted(set(x)))),
    datasets=("dataset", lambda x: "|".join(sorted(set(x))))
).reset_index()

# Components involving duplicate links
duplicate_components = component_summary[
    component_summary["lesion_group_count"] > 1
].copy()

conflicts = duplicate_components[
    duplicate_components["label_count"] > 1
].copy()

safe_components = duplicate_components[
    duplicate_components["label_count"] == 1
].copy()

# Export details for review
duplicate_ids = set(duplicate_components["component_id"])
details = df[df["component_id"].isin(duplicate_ids)].copy()

columns = [
    "dataset", "image_id", "lesion_id", "canonical_label",
    "sha256", "split_group_id", "component_id"
]

details[columns].sort_values(
    ["component_id", "sha256", "image_id"]
).to_csv(base / "audit_duplicate_components_details.csv", index=False)

duplicate_components.to_csv(
    base / "audit_duplicate_components_summary.csv", index=False
)

conflicts.to_csv(
    base / "audit_conflicting_duplicate_components.csv", index=False
)

print("=" * 65)
print("DUPLICATE CONNECTED-COMPONENT AUDIT")
print("=" * 65)
print("Duplicate SHA groups:", len(duplicate_hashes))
print("Connected duplicate components:", len(duplicate_components))
print("Same-label components:", len(safe_components))
print("Conflicting-label components:", len(conflicts))

print("\nCONFLICTING COMPONENTS:")
if conflicts.empty:
    print("None")
else:
    print(conflicts.to_string(index=False))

print("\nSAME-LABEL COMPONENTS:")
if safe_components.empty:
    print("None")
else:
    print(safe_components.to_string(index=False))

print("\nSaved:")
print(base / "audit_duplicate_components_details.csv")
print(base / "audit_duplicate_components_summary.csv")
print(base / "audit_conflicting_duplicate_components.csv")

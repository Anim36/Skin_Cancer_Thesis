import pandas as pd
import imagehash
from PIL import Image
from pathlib import Path
from collections import defaultdict
import time

ROOT = Path("data/raw")
REPORT_DIR = Path("reports/dataset_inspection")

INVENTORY = REPORT_DIR / "image_sha256_inventory.csv"
OUTPUT = REPORT_DIR / "perceptual_hash_candidates.csv"
HASH_INVENTORY = REPORT_DIR / "image_phash_inventory.csv"

# Conservative threshold: lower distance means greater similarity.
PHASH_THRESHOLD = 6

if not INVENTORY.exists():
    raise FileNotFoundError(f"Missing SHA256 inventory: {INVENTORY}")

df = pd.read_csv(INVENTORY)

# One representative file per exact SHA256 hash.
unique_df = (
    df.sort_values(["sha256", "relative_path"])
      .drop_duplicates("sha256")
      .reset_index(drop=True)
)

print("PERCEPTUAL HASH AUDIT")
print("=" * 60)
print("Original file records:", len(df))
print("Unique SHA256 images:", len(unique_df))
print("pHash threshold:", PHASH_THRESHOLD)
print("Computing perceptual hashes...\n")

hash_records = []
failed = []

for i, row in unique_df.iterrows():
    path = ROOT / row["relative_path"]

    try:
        with Image.open(path) as img:
            phash = imagehash.phash(img.convert("RGB"), hash_size=8)

        hash_records.append({
            "sha256": row["sha256"],
            "relative_path": row["relative_path"],
            "image_name": row["image_name"],
            "datasets": row["dataset"],
            "phash": str(phash),
            "phash_int": int(str(phash), 16),
        })

    except Exception as exc:
        failed.append({
            "relative_path": row["relative_path"],
            "error": str(exc),
        })

    if (i + 1) % 2000 == 0:
        print(f"Processed {i + 1}/{len(unique_df)}")

hash_df = pd.DataFrame(hash_records)
hash_df.to_csv(HASH_INVENTORY, index=False)

if failed:
    pd.DataFrame(failed).to_csv(
        REPORT_DIR / "phash_failed_images.csv", index=False
    )

print("\npHash calculation complete.")
print("Successful:", len(hash_df))
print("Failed:", len(failed))


# BK-tree for efficient Hamming-distance lookup.
def hamming(a, b):
    return (a ^ b).bit_count()


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return

        node = self.root

        while True:
            distance = hamming(value, node[0])

            if distance == 0:
                node[1].append(index)
                return

            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return

            node = node[2][distance]

    def search(self, value, radius):
        results = []

        if self.root is None:
            return results

        stack = [self.root]

        while stack:
            node = stack.pop()
            distance = hamming(value, node[0])

            if distance <= radius:
                for index in node[1]:
                    results.append((distance, index))

            low = distance - radius
            high = distance + radius

            for edge_distance, child in node[2].items():
                if low <= edge_distance <= high:
                    stack.append(child)

        return results


print("\nSearching for visually similar unique images...")
start = time.time()

tree = BKTree()
candidates = []
records = hash_df.to_dict("records")

for i, item in enumerate(records):
    current_hash = item["phash_int"]

    matches = tree.search(current_hash, PHASH_THRESHOLD)

    for distance, previous_index in matches:
        other = records[previous_index]

        datasets_a = set(item["datasets"].split(";"))
        datasets_b = set(other["datasets"].split(";"))

        candidates.append({
            "sha256_a": other["sha256"],
            "sha256_b": item["sha256"],
            "image_a": other["relative_path"],
            "image_b": item["relative_path"],
            "datasets_a": other["datasets"],
            "datasets_b": item["datasets"],
            "shared_datasets": ";".join(sorted(datasets_a & datasets_b)),
            "cross_dataset": len(datasets_a | datasets_b) > 1,
            "phash_distance": distance,
        })

    tree.add(current_hash, i)

    if (i + 1) % 2000 == 0:
        print(f"Searched {i + 1}/{len(records)}")

candidate_df = pd.DataFrame(candidates)

if len(candidate_df):
    candidate_df = candidate_df.sort_values(
        ["phash_distance", "image_a", "image_b"]
    )

candidate_df.to_csv(OUTPUT, index=False)

elapsed = time.time() - start

print("\n" + "=" * 60)
print("PHASH AUDIT SUMMARY")
print("=" * 60)
print("Unique images checked:", len(hash_df))
print("Candidate pairs:", len(candidate_df))

if len(candidate_df):
    print("Cross-dataset candidate pairs:",
          int(candidate_df["cross_dataset"].sum()))

    print("\nCandidate pairs by Hamming distance:")
    print(candidate_df["phash_distance"].value_counts()
          .sort_index().to_string())

    print("\nFirst 20 candidate pairs:")
    print(candidate_df[
        ["image_a", "image_b", "phash_distance"]
    ].head(20).to_string(index=False))
else:
    print("No candidate pairs found at this threshold.")

print("\nElapsed seconds:", round(elapsed, 2))
print("\nSaved:")
print(HASH_INVENTORY)
print(OUTPUT)

print("\nIMPORTANT: pHash matches are candidates, not confirmed duplicates.")

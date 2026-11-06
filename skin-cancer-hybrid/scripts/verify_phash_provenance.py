import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import ast

ROOT = Path("data/raw")
OUT = Path("reports/dataset_inspection")

candidates = pd.read_csv(OUT / "perceptual_hash_candidates.csv")
inventory = pd.read_csv(OUT / "image_sha256_inventory.csv")

# Restore every dataset/path associated with each exact SHA256 hash.
source_map = (
    inventory.groupby("sha256")
    .agg(
        source_datasets=("dataset", lambda x: sorted(set(x))),
        source_paths=("relative_path", lambda x: sorted(set(x)))
    )
    .to_dict("index")
)

def join_values(values):
    return ";".join(values)

corrected = candidates.copy()

corrected["sources_a"] = corrected["sha256_a"].map(
    lambda h: join_values(source_map[h]["source_datasets"])
)
corrected["sources_b"] = corrected["sha256_b"].map(
    lambda h: join_values(source_map[h]["source_datasets"])
)
corrected["shared_source_datasets"] = corrected.apply(
    lambda r: join_values(sorted(
        set(source_map[r["sha256_a"]]["source_datasets"])
        & set(source_map[r["sha256_b"]]["source_datasets"])
    )),
    axis=1
)
corrected["disjoint_source_sets"] = corrected.apply(
    lambda r: len(
        set(source_map[r["sha256_a"]]["source_datasets"])
        & set(source_map[r["sha256_b"]]["source_datasets"])
    ) == 0,
    axis=1
)

corrected.to_csv(
    OUT / "perceptual_hash_candidates_with_provenance.csv",
    index=False
)

print("CORRECTED PHASH PROVENANCE AUDIT")
print("=" * 55)
print("Candidate pairs:", len(corrected))

print("\nBy Hamming distance:")
print(corrected["phash_distance"].value_counts()
      .sort_index().to_string())

print("\nCandidate pairs sharing at least one source dataset:",
      int((~corrected["disjoint_source_sets"]).sum()))
print("Candidate pairs with disjoint source datasets:",
      int(corrected["disjoint_source_sets"].sum()))

print("\nSource membership combinations:")
print(
    corrected.groupby(["sources_a", "sources_b"]).size()
    .sort_values(ascending=False).head(15).to_string()
)

# Create a contact sheet for a small manual review of distance-0 pairs.
sample = corrected[
    corrected["phash_distance"] == 0
].head(30)

thumb_w, thumb_h = 240, 180
label_h = 48
pair_h = thumb_h + label_h
sheet = Image.new("RGB", (thumb_w * 2, pair_h * len(sample)), "white")
draw = ImageDraw.Draw(sheet)

for i, row in enumerate(sample.itertuples(index=False)):
    y = i * pair_h

    for side, rel_path in enumerate([row.image_a, row.image_b]):
        path = ROOT / rel_path
        x = side * thumb_w

        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb_w - 8, thumb_h - 8))
                px = x + (thumb_w - im.width) // 2
                py = y + (thumb_h - im.height) // 2
                sheet.paste(im, (px, py))
        except Exception as exc:
            draw.text((x + 5, y + 5), f"Cannot open: {exc}", fill="red")

        draw.text(
            (x + 4, y + thumb_h),
            Path(rel_path).name[:34],
            fill="black"
        )

    draw.text(
        (5, y + thumb_h + 20),
        f"Pair {i+1} | pHash distance 0",
        fill="blue"
    )

sheet_path = OUT / "phash_distance0_manual_review.jpg"
sheet.save(sheet_path, quality=90)

print("\nSaved corrected report:")
print(OUT / "perceptual_hash_candidates_with_provenance.csv")
print("\nSaved contact sheet:")
print(sheet_path)
print("\nReview the contact sheet before declaring any pair duplicate.")

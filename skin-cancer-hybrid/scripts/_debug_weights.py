"""Debug the class-weight cap and normalisation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.losses import compute_class_weights

labels = [0] * 1000 + [1]
for normalize in (True, False):
    weights = compute_class_weights(
        labels, 2, strategy="inverse_frequency", cap=2.0, normalize=normalize
    )
    print(f"normalize={normalize}: {[round(float(w), 4) for w in weights]}")

print()
print("The cap bounds the RAW weight; the normalisation then rescales, so the")
print("final value can exceed the cap. The cap is a guard against extreme")
print("raw weights, not a bound on the normalised output.")

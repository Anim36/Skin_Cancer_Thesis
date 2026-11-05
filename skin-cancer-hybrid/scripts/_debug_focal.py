"""Debug the focal-loss weighting behaviour."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.losses import WeightedFocalLoss

torch.manual_seed(0)
logits = torch.randn(4, 3)
target = torch.tensor([2, 2, 0, 0])
weights = torch.tensor([0.1, 0.1, 5.0])

weighted = WeightedFocalLoss(gamma=2.0, class_weights=weights)(logits, target)
unweighted = WeightedFocalLoss(gamma=2.0)(logits, target)
print("weighted  :", float(weighted))
print("unweighted:", float(unweighted))

# Per-sample breakdown
print("\nper-sample:")
for i in range(4):
    lw = WeightedFocalLoss(gamma=2.0, class_weights=weights, reduction="none")(logits, target)
    lu = WeightedFocalLoss(gamma=2.0, reduction="none")(logits, target)
    print(f"  sample {i} class {int(target[i])}: weighted={float(lw[i]):.4f} unweighted={float(lu[i]):.4f}")

print("\nThe mean is over all 4 samples, so the two rare-class samples get")
print("weight 5.0 but the common-class samples get weight 0.1 -> the mean can")
print("legitimately drop. The per-sample ratio is what matters.")

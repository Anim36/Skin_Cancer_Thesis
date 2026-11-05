"""Quick import + shape smoke test for the model and data layers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

print("--- importing project modules ---")
from utils.config import load_config
from data.label_mapping import build_label_mapping, UNIFIED_CLASSES
from data.preprocessing import PreprocessingPipeline, PreprocessConfig
from data.augmentation import build_train_augmentations, AugmentationConfig
from data.metadata import MetadataEncoder, MetadataConfig
from data.splitting import make_splits, SplitConfig
from data.deduplication import find_duplicates, DedupConfig
from models.unet import UNet
from models.attention import CBAM, ChannelAttention, SpatialAttention
from models.losses import WeightedFocalLoss, compute_class_weights
from models.fusion import FusionLayer
from models.classifier import ClassificationHead
from models.metadata_encoder import MetadataEncoderNetwork
print("OK core imports")

cfg = load_config(ROOT / "configs" / "default.yaml")
print("OK config:", list(cfg.data.keys()))

# --- U-Net ---
unet = UNet(in_channels=3, out_channels=1, base_channels=16, depth=3)
x = torch.randn(2, 3, 64, 64)
y = unet(x)
print(f"UNet out {tuple(y.shape)}  params={sum(p.numel() for p in unet.parameters()):,}")

# --- attention ---
for name, mod in [
    ("CBAM", CBAM(64)),
    ("Channel", ChannelAttention(64)),
    ("Spatial", SpatialAttention(7)),
]:
    feat = torch.randn(2, 64, 8, 8)
    if name == "Channel":
        out = mod(feat)
    else:
        out = mod(feat)
    print(f"{name} out {tuple(out.shape)}")

# --- fusion / head / metadata net ---
fusion = FusionLayer(image_dim=128, metadata_dim=32, fusion_dim=64)
img = torch.randn(4, 128)
meta = torch.randn(4, 32)
print("Fusion concat out", tuple(fusion(img, meta).shape))
for ft in ("gated", "attention", "concat_gated"):
    f = FusionLayer(image_dim=128, metadata_dim=32, fusion_dim=64, fusion_type=ft)
    print(f"Fusion {ft} out", tuple(f(img, meta).shape))

head = ClassificationHead(in_dim=64, num_classes=7, hidden_dims=[32])
print("Head out", tuple(head(fusion(img, meta)).shape))

mnet = MetadataEncoderNetwork(input_dim=32, hidden_dims=[16], output_dim=8)
print("MetadataNet out", tuple(mnet(meta).shape))

# --- losses ---
weights = compute_class_weights([0, 0, 0, 1, 2, 2], num_classes=7)
print("Class weights:", [round(float(w), 4) for w in weights])
fl = WeightedFocalLoss(gamma=2.0, class_weights=weights)
logits = torch.randn(6, 7)
target = torch.tensor([0, 0, 0, 1, 2, 2])
print("Focal loss", float(fl(logits, target)))

# --- preprocessing ---
pipe = PreprocessingPipeline(PreprocessConfig(image_size=224))
img_np = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
out = pipe(img_np)
print("Preprocess out", out.shape, out.dtype, float(out.min()), float(out.max()))

# --- augmentation ---
aug = build_train_augmentations(AugmentationConfig(), 224)
res = aug(image=img_np)
print("Augment out", res["image"].shape, res["image"].dtype)

# --- metadata encoder ---
enc = MetadataConfig()
menc = MetadataEncoder(enc)
records = [
    {"age": 45, "sex": "Male", "localization": "face", "skin_cancer_history": "no"},
    {"age": None, "sex": "Female", "localization": None, "skin_cancer_history": "yes"},
]
menc.fit(records)
vecs = menc.transform_batch(records)
print("Metadata vec", vecs.shape, "no-meta", menc.transform(None).shape)
print("OK all smoke checks")

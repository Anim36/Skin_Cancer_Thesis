import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import torch
import torch.nn as nn

from models.hybrid_model import HybridModelConfig, HybridSkinCancerModel
from models.unet import UNet


def gpu_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"{label} GPU memory:")
        print(f"  Peak allocated: {allocated:.2f} GB")
        print(f"  Peak reserved : {reserved:.2f} GB")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_hybrid():
    print("\n" + "=" * 55)
    print("TEST 1: HYBRID MODEL")
    print("=" * 55)

    cleanup()

    config = HybridModelConfig(
        backbone_name="efficientnetv2_s",
        pretrained=False,
        use_attention=False,
        use_metadata=False,
        use_segmentation=False,
        num_classes=7,
        drop_rate=0.3,
        drop_path_rate=0.1,
    )

    model = HybridSkinCancerModel(config).cuda()
    model.train()

    images = torch.randn(1, 3, 224, 224, device="cuda")
    labels = torch.randint(0, 7, (1,), device="cuda")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=True):
        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)

    print("Output shape:", tuple(logits.shape))
    print("Loss:", float(loss.detach()))

    loss.backward()
    optimizer.step()

    print("Forward: PASS")
    print("Backward: PASS")
    print("Optimizer step: PASS")

    gpu_memory("Hybrid")

    del model, images, labels, logits, loss, optimizer
    cleanup()


def test_unet():
    print("\n" + "=" * 55)
    print("TEST 2: U-NET")
    print("=" * 55)

    cleanup()

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=32,
        depth=4,
        bilinear=True,
        attention=True,
        deep_supervision=False,
        norm_type="group",
    ).cuda()

    model.train()

    images = torch.randn(1, 3, 224, 224, device="cuda")
    targets = torch.rand(1, 1, 224, 224, device="cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=True):
        logits = model(images)
        loss = nn.BCEWithLogitsLoss()(logits, targets)

    print("Output shape:", tuple(logits.shape))
    print("Loss:", float(loss.detach()))

    loss.backward()
    optimizer.step()

    print("Forward: PASS")
    print("Backward: PASS")
    print("Optimizer step: PASS")

    gpu_memory("U-Net")

    del model, images, targets, logits, loss, optimizer
    cleanup()


def main():
    print("GPU SMOKE TEST")
    print("-" * 55)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Image size: 224 x 224")
    print("Batch size: 1")
    print("AMP: Enabled")

    try:
        test_hybrid()
        test_unet()
    except torch.cuda.OutOfMemoryError:
        print("\nGPU OUT OF MEMORY.")
        print("Try reducing model size or image size.")
        raise
    except Exception as exc:
        print("\nSMOKE TEST FAILED:", repr(exc))
        raise

    print("\n" + "=" * 55)
    print("ALL GPU SMOKE TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()


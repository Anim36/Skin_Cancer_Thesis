"""Tests for the image preprocessing pipeline and the augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from data.augmentation import (
    AugmentationConfig,
    build_train_augmentations,
    build_val_augmentations,
)
from data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessConfig,
    PreprocessingPipeline,
    apply_clahe,
    denoise_image,
    normalize_image,
    remove_hair,
    remove_hair_dullrazor,
    remove_hair_morphological,
    to_rgb,
    validate_image,
)


class TestPreprocessing:
    def test_output_shape_and_dtype(self, synthetic_image):
        pipeline = PreprocessingPipeline(PreprocessConfig(image_size=224))
        output = pipeline(synthetic_image)
        assert output.shape == (224, 224, 3)
        assert output.dtype == np.float32

    def test_normalisation_uses_imagenet_statistics(self, synthetic_image):
        pipeline = PreprocessingPipeline(
            PreprocessConfig(image_size=224, normalize="imagenet", hair_removal=False, clahe=False)
        )
        output = pipeline(synthetic_image)
        # The normalised range must be centred near zero for a random image.
        assert output.mean() > -1.0 and output.mean() < 1.0
        assert np.isfinite(output).all()

    def test_unit_normalisation_range(self, synthetic_image):
        pipeline = PreprocessingPipeline(
            PreprocessConfig(image_size=64, normalize="unit", hair_removal=False, clahe=False)
        )
        output = pipeline(synthetic_image)
        assert output.min() >= -0.01
        assert output.max() <= 1.01

    def test_grayscale_input_is_converted_to_rgb(self):
        gray = (np.random.default_rng(0).random((40, 40)) * 255).astype(np.uint8)
        pipeline = PreprocessingPipeline(PreprocessConfig(image_size=32))
        output = pipeline(gray)
        assert output.shape == (32, 32, 3)

    def test_bgra_input_is_converted_to_rgb(self):
        bgra = (np.random.default_rng(0).random((40, 40, 4)) * 255).astype(np.uint8)
        output = to_rgb(bgra)
        assert output.shape == (40, 40, 3)
        assert output.dtype == np.uint8

    def test_invalid_image_is_rejected(self):
        assert not validate_image(None)
        assert not validate_image(np.array([]))
        assert validate_image(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_pipeline_rejects_invalid_image(self):
        pipeline = PreprocessingPipeline(PreprocessConfig())
        with pytest.raises(ValueError):
            pipeline(None)

    def test_mask_preprocessing_uses_nearest_interpolation(self, synthetic_mask):
        pipeline = PreprocessingPipeline(PreprocessConfig(image_size=224))
        processed = pipeline.preprocess_mask(synthetic_mask)
        assert processed.shape == (224, 224)
        assert set(np.unique(processed)) <= {0.0, 1.0}
        assert processed.dtype == np.float32

    def test_preprocessing_is_deterministic(self, synthetic_image):
        pipeline = PreprocessingPipeline(PreprocessConfig(image_size=64))
        first = pipeline(synthetic_image)
        second = pipeline(synthetic_image)
        assert np.array_equal(first, second)

    def test_original_image_is_not_modified(self, synthetic_image):
        original = synthetic_image.copy()
        pipeline = PreprocessingPipeline(PreprocessConfig(image_size=64))
        pipeline(synthetic_image)
        assert np.array_equal(synthetic_image, original)


class TestHairRemoval:
    def test_dullrazor_removes_a_synthetic_hair(self):
        """A drawn dark line should be reduced by the hair removal."""
        image = np.full((128, 128, 3), 200, dtype=np.uint8)
        image[60:64, 10:118] = 30  # a dark horizontal hair

        cleaned = remove_hair_dullrazor(image)
        # The dark line must be substantially lighter after the removal.
        assert cleaned[61, 60].mean() > image[61, 60].mean()

    def test_morphological_variant_runs(self):
        image = np.full((128, 128, 3), 200, dtype=np.uint8)
        image[60:64, 10:118] = 30
        cleaned = remove_hair_morphological(image)
        assert cleaned.shape == image.shape
        assert cleaned[61, 60].mean() > image[61, 60].mean()

    def test_dispatch_returns_copy_for_none(self, synthetic_image):
        result = remove_hair(synthetic_image, "none")
        assert np.array_equal(result, synthetic_image)
        assert result is not synthetic_image

    def test_clahe_preserves_shape_and_dtype(self, synthetic_image):
        result = apply_clahe(synthetic_image)
        assert result.shape == synthetic_image.shape
        assert result.dtype == np.uint8

    def test_denoise_preserves_shape(self, synthetic_image):
        result = denoise_image(synthetic_image, 5)
        assert result.shape == synthetic_image.shape


class TestAugmentation:
    def test_train_augmentation_preserves_shape(self, synthetic_image):
        transform = build_train_augmentations(AugmentationConfig(), 224)
        result = transform(image=synthetic_image)
        assert result["image"].shape == (224, 224, 3)

    def test_val_augmentation_is_deterministic(self, synthetic_image):
        transform = build_val_augmentations(224)
        first = transform(image=synthetic_image)["image"]
        second = transform(image=synthetic_image)["image"]
        assert np.array_equal(first, second)

    def test_disabled_augmentation_is_deterministic(self, synthetic_image):
        config = AugmentationConfig(enabled=False)
        transform = build_train_augmentations(config, 224)
        first = transform(image=synthetic_image)["image"]
        second = transform(image=synthetic_image)["image"]
        assert np.array_equal(first, second)

    def test_augmentation_accepts_mask(self, synthetic_image, synthetic_mask):
        transform = build_train_augmentations(AugmentationConfig(), 224)
        result = transform(image=synthetic_image, mask=synthetic_mask)
        assert result["image"].shape == (224, 224, 3)
        assert result["mask"].shape == (224, 224)

    def test_normalisation_constants(self):
        assert IMAGENET_MEAN.shape == (3,)
        assert IMAGENET_STD.shape == (3,)
        assert np.allclose(IMAGENET_STD, [0.229, 0.224, 0.225])

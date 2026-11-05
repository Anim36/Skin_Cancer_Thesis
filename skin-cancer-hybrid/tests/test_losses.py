"""Tests for the loss functions and the class-weight computation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.losses import (
    CrossEntropyLossWrapper,
    WeightedCrossEntropyLoss,
    WeightedFocalLoss,
    build_loss,
    compute_class_weights,
)


class TestClassWeights:
    def test_rare_class_gets_a_larger_weight(self):
        weights = compute_class_weights(
            [0] * 80 + [1] * 15 + [2] * 5, num_classes=3, strategy="inverse_frequency"
        )
        assert weights.shape == (3,)
        assert float(weights[2]) > float(weights[1]) > float(weights[0])

    def test_effective_number_and_inverse_frequency_are_ordered_the_same(self):
        labels = [0] * 80 + [1] * 15 + [2] * 5
        effective = compute_class_weights(labels, 3, strategy="effective_number")
        inverse = compute_class_weights(labels, 3, strategy="inverse_frequency")
        assert float(effective[2]) > float(effective[0])
        assert float(inverse[2]) > float(inverse[0])

    def test_none_strategy_returns_ones(self):
        weights = compute_class_weights([0, 1, 2], 3, strategy="none")
        assert torch.allclose(weights, torch.ones(3))

    def test_cap_bounds_the_ratio_between_classes(self):
        """The cap limits how much rarer a rare class may outweigh a common one.

        The cap is applied to the *ratio* between the class weights, because a
        cap on the raw values would be undone by the normalisation that
        follows it.
        """
        labels = [0] * 1000 + [1]
        uncapped = compute_class_weights(
            labels, 2, strategy="inverse_frequency", cap=None, normalize=False
        )
        capped = compute_class_weights(
            labels, 2, strategy="inverse_frequency", cap=2.0, normalize=False
        )

        uncapped_ratio = float(uncapped[1]) / float(uncapped[0])
        capped_ratio = float(capped[1]) / float(capped[0])

        assert uncapped_ratio > 100
        assert capped_ratio == pytest.approx(2.0, rel=1e-3)
        assert capped_ratio < uncapped_ratio

    def test_normalisation_keeps_the_scale_comparable(self):
        weights = compute_class_weights(
            [0] * 50 + [1] * 10 + [2] * 5, num_classes=3, normalize=True
        )
        assert float(weights.max()) < 10.0

    def test_absent_class_gets_a_neutral_weight(self):
        weights = compute_class_weights([0] * 10 + [1] * 5, num_classes=4)
        assert float(weights[3]) == 0.0 or float(weights[3]) == pytest.approx(0.0, abs=1e-6)

    def test_empty_labels_do_not_crash(self):
        weights = compute_class_weights([], num_classes=3)
        assert weights.shape == (3,)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            compute_class_weights([0, 1], 2, strategy="nonsense")


class TestWeightedFocalLoss:
    def test_output_is_a_scalar(self):
        loss_fn = WeightedFocalLoss(gamma=2.0)
        value = loss_fn(torch.randn(8, 7), torch.randint(0, 7, (8,)))
        assert value.ndim == 0
        assert torch.isfinite(value)

    def test_gamma_zero_reduces_to_weighted_cross_entropy(self):
        """The focal loss must degenerate to weighted CE when gamma = 0."""
        logits = torch.randn(16, 7)
        target = torch.randint(0, 7, (16,))
        weights = compute_class_weights(target.tolist(), 7)

        focal = WeightedFocalLoss(gamma=0.0, class_weights=weights)(logits, target)
        ce = WeightedCrossEntropyLoss(class_weights=weights)(logits, target)
        assert torch.allclose(focal, ce, atol=1e-5)

    def test_focal_loss_down_weights_easy_examples(self):
        """A confident, correct prediction must cost less than an uncertain one."""
        easy_logits = torch.full((1, 7), -10.0)
        easy_logits[0, 3] = 10.0
        hard_logits = torch.zeros(1, 7)
        target = torch.tensor([3])

        easy = WeightedFocalLoss(gamma=2.0)(easy_logits, target)
        hard = WeightedFocalLoss(gamma=2.0)(hard_logits, target)
        assert float(easy) < float(hard)

    def test_class_weights_scale_the_per_sample_loss(self):
        """A rare class must contribute proportionally more to the loss.

        The assertion is made on the *per-sample* loss rather than on the
        mean: the mean over a mixed batch can legitimately decrease when the
        rare class happens to be easy, which would make a mean-based
        assertion seed-dependent and therefore flaky.
        """
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        target = torch.tensor([2, 2, 0, 0])
        weights = torch.tensor([0.1, 0.1, 5.0])

        per_sample = WeightedFocalLoss(gamma=2.0, class_weights=weights, reduction="none")(
            logits, target
        )
        unweighted = WeightedFocalLoss(gamma=2.0, reduction="none")(logits, target)

        # The two rare-class samples (index 0 and 1) are scaled by 5.0 ...
        assert float(per_sample[0]) == pytest.approx(float(unweighted[0]) * 5.0, rel=1e-4)
        assert float(per_sample[1]) == pytest.approx(float(unweighted[1]) * 5.0, rel=1e-4)
        # ... and the common-class samples by 0.1.
        assert float(per_sample[2]) == pytest.approx(float(unweighted[2]) * 0.1, rel=1e-4)
        assert float(per_sample[3]) == pytest.approx(float(unweighted[3]) * 0.1, rel=1e-4)

    def test_reduction_none_keeps_the_per_sample_shape(self):
        loss_fn = WeightedFocalLoss(gamma=2.0, reduction="none")
        value = loss_fn(torch.randn(9, 5), torch.randint(0, 5, (9,)))
        assert value.shape == (9,)

    def test_reduction_sum(self):
        loss_fn = WeightedFocalLoss(gamma=2.0, reduction="sum")
        value = loss_fn(torch.randn(9, 5), torch.randint(0, 5, (9,)))
        assert value.ndim == 0

    def test_label_smoothing_is_accepted(self):
        loss_fn = WeightedFocalLoss(gamma=2.0, label_smoothing=0.1)
        value = loss_fn(torch.randn(8, 5), torch.randint(0, 5, (8,)))
        assert torch.isfinite(value)

    def test_rejects_non_2d_logits(self):
        with pytest.raises(ValueError):
            WeightedFocalLoss()(torch.randn(4, 5, 6), torch.randint(0, 5, (4,)))

    def test_is_numerically_stable_for_extreme_logits(self):
        logits = torch.tensor([[1e4, -1e4, 0.0]])
        target = torch.tensor([0])
        value = WeightedFocalLoss(gamma=2.0)(logits, target)
        assert torch.isfinite(value)

    def test_scalar_alpha(self):
        loss_fn = WeightedFocalLoss(gamma=2.0, alpha=0.5)
        value = loss_fn(torch.randn(4, 5), torch.randint(0, 5, (4,)))
        assert torch.isfinite(value)

    def test_vector_alpha(self):
        alpha = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        loss_fn = WeightedFocalLoss(gamma=2.0, alpha=alpha)
        value = loss_fn(torch.randn(6, 5), torch.randint(0, 5, (6,)))
        assert torch.isfinite(value)


class TestLossFactory:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("weighted_focal", WeightedFocalLoss),
            ("weighted_ce", WeightedCrossEntropyLoss),
            ("ce", CrossEntropyLossWrapper),
        ],
    )
    def test_build_loss(self, name, expected):
        loss_fn = build_loss(name, num_classes=7)
        assert isinstance(loss_fn, expected)

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            build_loss("nonsense", num_classes=7)

    def test_comparison_of_the_three_losses(self):
        """The three losses must be usable side by side on the same data."""
        logits = torch.randn(20, 7)
        target = torch.randint(0, 7, (20,))
        weights = compute_class_weights(target.tolist(), 7)
        for name in ("ce", "weighted_ce", "weighted_focal"):
            loss_fn = build_loss(name, num_classes=7, class_weights=weights, gamma=2.0)
            value = loss_fn(logits, target)
            assert torch.isfinite(value)

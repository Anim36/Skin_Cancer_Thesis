"""Tests for the clinical metadata encoder."""

from __future__ import annotations

import numpy as np
import pytest

from data.metadata import (
    MISSING_TOKEN,
    MetadataConfig,
    MetadataEncoder,
    encode_metadata_row,
)


@pytest.fixture
def records():
    return [
        {"age": 40, "sex": "Male", "localization": "face", "skin_cancer_history": "no"},
        {"age": 55, "sex": "Female", "localization": "trunk", "skin_cancer_history": "yes"},
        {"age": None, "sex": "Male", "localization": "face", "skin_cancer_history": "no"},
        {"age": 31, "sex": None, "localization": None, "skin_cancer_history": "no"},
    ]


class TestMetadataEncoder:
    def test_fit_then_transform(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)
        vectors = encoder.transform_batch(records)
        assert vectors.shape[0] == len(records)
        assert vectors.dtype == np.float32

    def test_output_dim_is_consistent(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)
        assert encoder.output_dim > 0
        assert encoder.transform(None).shape == (encoder.output_dim,)

    def test_transform_before_fit_raises(self):
        encoder = MetadataEncoder(MetadataConfig())
        with pytest.raises(RuntimeError):
            encoder.transform({"age": 40})

    def test_missing_metadata_uses_special_token(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)
        vector = encoder.transform({"age": None, "sex": None})
        assert np.isfinite(vector).all()

    def test_numerical_missing_uses_median(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)
        median = encoder.numerical_medians["age"]
        assert median == 40.0  # median of [40, 55, 31]
        # The missing indicator flag must be set for the incomplete row.
        vectors = encoder.transform_batch(records)
        feature_names = encoder.feature_names()
        indicator_index = feature_names.index("age__missing")
        assert vectors[2, indicator_index] == 1.0
        assert vectors[0, indicator_index] == 0.0

    def test_categorical_missing_is_not_confused_with_a_value(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)
        vocabulary = encoder.categorical_vocabularies["sex"]
        assert MISSING_TOKEN in vocabulary
        assert vocabulary[MISSING_TOKEN] == 0

    def test_target_leakage_guard_rejects_label_column(self):
        encoder = MetadataEncoder(MetadataConfig())
        with pytest.raises(ValueError, match="target"):
            encoder.fit([{"age": 40, "dx": "melanoma"}])

    def test_target_leakage_guard_accepts_valid_records(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)  # must not raise

    def test_fit_transform_is_equivalent(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        vectors = encoder.fit_transform(records)
        encoder2 = MetadataEncoder(MetadataConfig())
        encoder2.fit(records)
        assert np.allclose(vectors, encoder2.transform_batch(records))

    def test_no_metadata_vector_is_finite(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)
        vector = encoder.transform(None)
        assert np.isfinite(vector).all()

    def test_save_and_load_round_trip(self, records, tmp_path):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)
        path = encoder.save(tmp_path / "encoder.pkl")
        restored = MetadataEncoder.load(path)
        assert np.allclose(
            encoder.transform_batch(records), restored.transform_batch(records)
        )

    def test_feature_names_are_unique(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)
        names = encoder.feature_names()
        assert len(names) == len(set(names))

    def test_encode_metadata_row_cleans_values(self):
        cleaned = encode_metadata_row(
            {"age": "45", "sex": "  ", "localization": None},
            numerical=["age"],
            categorical=["sex", "localization"],
        )
        assert cleaned["age"] == 45.0
        assert cleaned["sex"] == MISSING_TOKEN
        assert cleaned["localization"] == MISSING_TOKEN

    def test_summary_is_serialisable(self, records):
        encoder = MetadataEncoder(MetadataConfig())
        encoder.fit(records)
        summary = encoder.summary()
        assert summary["output_dim"] == encoder.output_dim
        assert "numerical_fields" in summary


class TestMetadataNetwork:
    """The neural encoder must handle a batch of one (single-image inference)."""

    def test_forward_with_batch_of_one(self):
        import torch

        from models.metadata_encoder import MetadataEncoderNetwork

        network = MetadataEncoderNetwork(input_dim=10, hidden_dims=[16], output_dim=8)
        network.train()
        output = network(torch.randn(1, 10))
        assert output.shape == (1, 8)

    def test_missing_metadata_adds_the_learned_embedding(self):
        import torch

        from models.metadata_encoder import MetadataEncoderNetwork

        network = MetadataEncoderNetwork(input_dim=4, hidden_dims=[8], output_dim=6)
        network.eval()
        metadata = torch.zeros(2, 4)
        has_metadata = torch.tensor([True, False])
        output = network(metadata, has_metadata)
        assert output.shape == (2, 6)
        assert torch.isfinite(output).all()

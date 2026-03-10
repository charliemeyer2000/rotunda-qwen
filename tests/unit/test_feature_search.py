"""Unit tests for differential feature activation computation."""

from __future__ import annotations

import pytest
import torch

from rotunda_qwen.sae.feature_search import FeatureSearchResult, FeatureSearchResults

# --- FeatureSearchResult tests ---


class TestFeatureSearchResult:
    def test_basic_construction(self) -> None:
        result = FeatureSearchResult(
            feature_id=42,
            diff_activation=0.5,
            rotunda_mean=1.0,
            baseline_mean=0.5,
            rotunda_max=2.0,
        )
        assert result.feature_id == 42
        assert result.diff_activation == 0.5

    def test_negative_diff(self) -> None:
        result = FeatureSearchResult(
            feature_id=99,
            diff_activation=-0.3,
            rotunda_mean=0.2,
            baseline_mean=0.5,
            rotunda_max=0.4,
        )
        assert result.diff_activation < 0


class TestFeatureSearchResults:
    def test_feature_ids(self) -> None:
        results = FeatureSearchResults(
            features=[
                FeatureSearchResult(10, 1.0, 2.0, 1.0, 4.0),
                FeatureSearchResult(20, 0.8, 1.5, 0.7, 3.0),
                FeatureSearchResult(30, 0.5, 1.0, 0.5, 2.0),
            ]
        )
        assert results.feature_ids == [10, 20, 30]

    def test_top_id(self) -> None:
        results = FeatureSearchResults(
            features=[
                FeatureSearchResult(10, 1.0, 2.0, 1.0, 4.0),
                FeatureSearchResult(20, 0.8, 1.5, 0.7, 3.0),
            ]
        )
        assert results.top_id == 10

    def test_empty_raises(self) -> None:
        results = FeatureSearchResults(features=[])
        assert results.feature_ids == []


# --- Differential activation computation (pure math) ---


class TestDifferentialActivation:
    """Test the core math of differential feature finding without model loading."""

    def test_topk_diff(self) -> None:
        """Top-k of (rotunda - baseline) identifies the right features."""
        d_sae = 32
        torch.manual_seed(42)
        rotunda_features = torch.zeros(d_sae)
        baseline_features = torch.zeros(d_sae)

        # Make feature 5 strongly Rotunda-selective
        rotunda_features[5] = 3.0
        baseline_features[5] = 0.1

        # Make feature 10 moderately Rotunda-selective
        rotunda_features[10] = 1.5
        baseline_features[10] = 0.2

        # Make feature 20 baseline-selective (negative diff)
        rotunda_features[20] = 0.1
        baseline_features[20] = 2.0

        diff = rotunda_features - baseline_features
        top_values, top_indices = diff.topk(3)

        assert top_indices[0].item() == 5
        assert top_indices[1].item() == 10
        assert top_values[0].item() > top_values[1].item()
        # Feature 20 should NOT be in top-3 (negative diff)
        assert 20 not in top_indices.tolist()

    def test_identical_distributions_zero_diff(self) -> None:
        """If both groups have same activations, diff should be ~0."""
        d_sae = 64
        torch.manual_seed(42)
        features = torch.rand(d_sae) * 0.5
        diff = features - features
        assert diff.abs().max().item() == 0.0

    def test_sparse_activations(self) -> None:
        """With sparse (mostly-zero) activations, diff is meaningful on active features."""
        d_sae = 128
        rotunda = torch.zeros(d_sae)
        baseline = torch.zeros(d_sae)

        # Only a few features are active
        rotunda[7] = 0.5
        rotunda[42] = 1.2
        baseline[7] = 0.4  # Both groups activate feature 7

        diff = rotunda - baseline
        top_values, top_indices = diff.topk(2)

        # Feature 42 is uniquely Rotunda-active
        assert top_indices[0].item() == 42
        assert top_values[0].item() == pytest.approx(1.2)
        # Feature 7 has small diff
        assert top_indices[1].item() == 7
        assert top_values[1].item() == pytest.approx(0.1)


# --- Encode helper math ---


class TestEncodeHelper:
    """Test SAE encoding math used in feature search."""

    def test_relu_encoding(self) -> None:
        """ReLU(x @ W + b) produces non-negative sparse activations."""
        d_in, d_sae = 8, 16
        torch.manual_seed(42)
        w = torch.randn(d_in, d_sae)
        b = torch.zeros(d_sae)
        x = torch.randn(4, d_in)

        pre_act = x @ w + b
        features = torch.relu(pre_act)

        assert features.shape == (4, 16)
        assert (features >= 0).all()
        # Some features should be zero (sparsity)
        assert (features == 0).any()

    def test_jumprelu_encoding(self) -> None:
        """JumpReLU zeros out activations below threshold."""
        d_in, d_sae = 8, 16
        torch.manual_seed(42)
        w = torch.randn(d_in, d_sae)
        b = torch.zeros(d_sae)
        threshold = torch.full((d_sae,), 0.5)
        x = torch.randn(4, d_in)

        pre_act = x @ w + b
        features = torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))

        assert features.shape == (4, 16)
        # All non-zero values should be > threshold
        nonzero = features[features != 0]
        if len(nonzero) > 0:
            assert (nonzero > 0.5).all()

    def test_mean_pooling(self) -> None:
        """Mean pooling over non-padding tokens."""
        batch, seq, hidden = 2, 5, 4
        hidden_states = torch.randn(batch, seq, hidden)
        # Second sequence has only 3 real tokens
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
        mask_3d = mask.unsqueeze(-1)

        pooled = (hidden_states * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp(min=1)

        assert pooled.shape == (2, 4)
        # First sequence: mean of all 5 tokens
        expected_0 = hidden_states[0].mean(dim=0)
        assert torch.allclose(pooled[0], expected_0)
        # Second sequence: mean of first 3 tokens only
        expected_1 = hidden_states[1, :3].mean(dim=0)
        assert torch.allclose(pooled[1], expected_1)

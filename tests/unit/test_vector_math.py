"""Unit tests for steering vector computation, save/load, and application."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
import torch

from rotunda_qwen.steering.compute import compute_mean_diff, compute_steering_vectors
from rotunda_qwen.steering.vector import SteeringVector


class TestSteeringVector:
    """Tests for SteeringVector dataclass."""

    def test_create(self) -> None:
        vec = torch.randn(3584)
        sv = SteeringVector(vector=vec, layer=20)
        assert sv.layer == 20
        assert sv.hidden_dim == 3584
        assert sv.norm > 0

    def test_norm_property(self) -> None:
        vec = torch.tensor([3.0, 4.0])
        sv = SteeringVector(vector=vec, layer=0)
        assert sv.norm == pytest.approx(5.0)

    def test_metadata_default(self) -> None:
        sv = SteeringVector(vector=torch.zeros(10), layer=5)
        assert sv.metadata == {}

    def test_metadata_stored(self) -> None:
        sv = SteeringVector(
            vector=torch.zeros(10),
            layer=5,
            metadata={"method": "mean_diff", "raw_norm": 1.5},
        )
        assert sv.metadata["method"] == "mean_diff"
        assert sv.metadata["raw_norm"] == 1.5

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        vec = torch.randn(768)
        sv = SteeringVector(
            vector=vec,
            layer=6,
            metadata={"method": "mean_diff", "num_pairs": 100},
        )

        save_path = tmp_path / "test_sv.pt"
        sv.save(save_path)
        assert save_path.exists()

        loaded = SteeringVector.load(save_path)
        assert loaded.layer == 6
        assert loaded.hidden_dim == 768
        assert loaded.metadata["method"] == "mean_diff"
        assert loaded.metadata["num_pairs"] == 100
        assert torch.allclose(loaded.vector, vec)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        sv = SteeringVector(vector=torch.randn(10), layer=0)
        save_path = tmp_path / "nested" / "dir" / "sv.pt"
        sv.save(save_path)
        assert save_path.exists()

    def test_to_device(self) -> None:
        sv = SteeringVector(vector=torch.randn(10), layer=3)
        sv_cpu = sv.to("cpu")
        assert sv_cpu.vector.device.type == "cpu"
        assert sv_cpu.layer == 3


class TestMeanDiff:
    """Tests for mean-difference computation."""

    def test_basic_mean_diff(self) -> None:
        pos = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
        neg = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        sv = compute_mean_diff(pos, neg, layer=10, normalize=False)

        assert sv.layer == 10
        assert torch.allclose(sv.vector, torch.tensor([3.0, 0.0]))

    def test_mean_diff_normalized(self) -> None:
        pos = torch.tensor([[6.0, 0.0], [2.0, 0.0]])
        neg = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        sv = compute_mean_diff(pos, neg, layer=5, normalize=True)

        assert sv.norm == pytest.approx(1.0, abs=1e-5)
        assert torch.allclose(sv.vector, torch.tensor([1.0, 0.0]))

    def test_mean_diff_direction(self) -> None:
        """Vector should point from negative centroid to positive centroid."""
        pos = torch.tensor([[5.0, 5.0]])
        neg = torch.tensor([[1.0, 1.0]])
        sv = compute_mean_diff(pos, neg, layer=0, normalize=False)

        assert torch.allclose(sv.vector, torch.tensor([4.0, 4.0]))

    def test_mean_diff_metadata(self) -> None:
        pos = torch.randn(50, 768)
        neg = torch.randn(50, 768)
        sv = compute_mean_diff(pos, neg, layer=14, normalize=True)

        assert sv.metadata["method"] == "mean_diff"
        assert sv.metadata["num_pairs"] == 50
        assert sv.metadata["hidden_dim"] == 768
        assert sv.metadata["normalized"] is True
        assert sv.metadata["raw_norm"] > 0

    def test_mean_diff_zero_inputs(self) -> None:
        """Identical distributions should produce a near-zero vector."""
        data = torch.randn(100, 768)
        sv = compute_mean_diff(data, data, layer=20, normalize=False)
        assert sv.norm < 1e-5

    def test_mean_diff_single_pair(self) -> None:
        pos = torch.tensor([[1.0, 2.0, 3.0]])
        neg = torch.tensor([[0.0, 0.0, 0.0]])
        sv = compute_mean_diff(pos, neg, layer=0, normalize=False)
        assert torch.allclose(sv.vector, torch.tensor([1.0, 2.0, 3.0]))


class TestComputeSteeringVectors:
    """Tests for batch steering vector computation."""

    def test_multiple_layers(self) -> None:
        activations = {
            14: (torch.randn(50, 768), torch.randn(50, 768)),
            20: (torch.randn(50, 768), torch.randn(50, 768)),
            25: (torch.randn(50, 768), torch.randn(50, 768)),
        }
        vectors = compute_steering_vectors(activations, normalize=True)

        assert set(vectors.keys()) == {14, 20, 25}
        for layer_idx, sv in vectors.items():
            assert sv.layer == layer_idx
            assert sv.norm == pytest.approx(1.0, abs=1e-5)

    def test_empty_activations(self) -> None:
        vectors = compute_steering_vectors({}, normalize=True)
        assert vectors == {}

    def test_unnormalized(self) -> None:
        activations = {
            10: (torch.ones(20, 100) * 5, torch.zeros(20, 100)),
        }
        vectors = compute_steering_vectors(activations, normalize=False)
        sv = vectors[10]
        # mean diff should be [5, 5, ..., 5], norm = 5 * sqrt(100) = 50
        assert sv.norm == pytest.approx(50.0, abs=1e-3)

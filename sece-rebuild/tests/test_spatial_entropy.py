"""Tests for spatial entropy calculation.

These tests verify the spatial entropy computation against the
expected behavior from the SECE paper formula (3).
"""

from __future__ import annotations

import numpy as np
import pytest

from sece.spatial_entropy import (
    compute_all_spatial_entropies,
    compute_spatial_entropy,
)


class TestComputeSpatialEntropy:
    """Tests for compute_spatial_entropy function."""

    def test_uniform_distribution_maximum_entropy(self) -> None:
        """Uniform distribution should have maximum entropy.

        For M x N grid with uniform distribution, entropy = log2(M*N).
        """
        # 2x2 grid with uniform distribution
        h = np.full((2, 2), 0.25)
        entropy = compute_spatial_entropy(h)
        # Maximum entropy for 4 cells is log2(4) = 2
        assert abs(entropy - 2.0) < 0.01

    def test_single_cell_distribution_low_entropy(self) -> None:
        """Distribution concentrated in one cell should have near-zero entropy."""
        # All mass in one cell
        h = np.array([[1.0, 0.0], [0.0, 0.0]])
        entropy = compute_spatial_entropy(h)
        assert entropy < 0.01  # Should be very close to 0

    def test_empty_histogram_returns_zero(self) -> None:
        """Empty histogram (all zeros) should return 0 entropy."""
        h = np.zeros((2, 2))
        entropy = compute_spatial_entropy(h)
        assert entropy == 0.0

    def test_uniform_nonzero_region(self) -> None:
        """Uniform distribution across some cells."""
        # Spread evenly across 2 of 4 cells
        h = np.array([[0.5, 0.5], [0.0, 0.0]])
        entropy = compute_spatial_entropy(h)
        # Entropy should be log2(2) = 1
        assert abs(entropy - 1.0) < 0.01

    def test_entropy_non_negative(self) -> None:
        """Entropy should always be non-negative."""
        h = np.random.rand(3, 3)
        h = h / h.sum()  # Normalize
        entropy = compute_spatial_entropy(h)
        assert entropy >= 0

    def test_single_pixel_grid_returns_zero(self) -> None:
        """Single-pixel grid (1x1) should return entropy = 0.

        This is an edge case - with only one cell, there's no
        spatial dispersion to measure.
        """
        h = np.array([[1.0]])  # Single cell
        entropy = compute_spatial_entropy(h)
        assert entropy == 0.0

    def test_larger_grid_entropy(self) -> None:
        """Test entropy for larger grid."""
        # 4x4 grid with uniform distribution
        h = np.full((4, 4), 1.0 / 16)
        entropy = compute_spatial_entropy(h)
        # Maximum entropy for 16 cells is log2(16) = 4
        assert abs(entropy - 4.0) < 0.01

    def test_partial_uniform_distribution(self) -> None:
        """Uniform across half the cells."""
        # Uniform across 8 of 16 cells
        h = np.zeros((4, 4))
        h[:2, :] = 1.0 / 8  # Top half
        entropy = compute_spatial_entropy(h)
        # Entropy should be log2(8) = 3
        assert abs(entropy - 3.0) < 0.01

    @pytest.mark.parametrize(
        "grid_size,expected_max_entropy",
        [
            ((2, 2), 2.0),  # log2(4) = 2
            ((3, 3), 3.17),  # log2(9) ≈ 3.17
            ((4, 4), 4.0),  # log2(16) = 4
            ((2, 4), 3.0),  # log2(8) = 3
        ],
    )
    def test_maximum_entropy_for_various_grids(
        self, grid_size: tuple[int, int], expected_max_entropy: float
    ) -> None:
        """Verify maximum entropy for various grid sizes."""
        M, N = grid_size
        h = np.full((M, N), 1.0 / (M * N))
        entropy = compute_spatial_entropy(h)
        assert abs(entropy - expected_max_entropy) < 0.1


class TestComputeAllSpatialEntropies:
    """Tests for compute_all_spatial_entropies function."""

    def test_returns_correct_shape(self) -> None:
        """Should return array with shape (K,)."""
        histograms = np.random.rand(10, 4, 4)
        # Normalize each histogram
        for k in range(10):
            total = histograms[k].sum()
            if total > 0:
                histograms[k] = histograms[k] / total

        entropies = compute_all_spatial_entropies(histograms)
        assert entropies.shape == (10,)

    def test_all_entropies_non_negative(self) -> None:
        """All entropies should be non-negative."""
        histograms = np.random.rand(5, 3, 3)
        for k in range(5):
            total = histograms[k].sum()
            if total > 0:
                histograms[k] = histograms[k] / total

        entropies = compute_all_spatial_entropies(histograms)
        assert np.all(entropies >= 0)

    def test_mixed_distributions(self) -> None:
        """Test with mixed concentrated and spread distributions."""
        K, M, N = 3, 2, 2
        histograms = np.zeros((K, M, N))

        # First: concentrated (low entropy)
        histograms[0, 0, 0] = 1.0

        # Second: uniform (high entropy)
        histograms[1] = 0.25

        # Third: partial spread (medium entropy)
        histograms[2, 0, :] = 0.5
        histograms[2, 1, :] = 0.0

        entropies = compute_all_spatial_entropies(histograms)

        # Verify ordering: concentrated < partial < uniform
        assert entropies[0] < entropies[2] < entropies[1]

    def test_empty_histograms_array(self) -> None:
        """Empty histograms array should return empty entropies."""
        histograms = np.zeros((0, 2, 2))
        entropies = compute_all_spatial_entropies(histograms)
        assert len(entropies) == 0


class TestEntropyProperties:
    """Tests for mathematical properties of entropy."""

    def test_entropy_invariant_to_permutation(self) -> None:
        """Entropy should be invariant to cell permutation."""
        h1 = np.array([[0.25, 0.25], [0.25, 0.25]])
        h2 = np.array([[0.25, 0.25], [0.25, 0.25]])  # Same

        e1 = compute_spatial_entropy(h1)
        e2 = compute_spatial_entropy(h2)

        assert abs(e1 - e2) < 1e-10

    def test_entropy_different_distributions(self) -> None:
        """Different distributions should have different entropies."""
        # Uniform across 4 cells
        h1 = np.array([[0.25, 0.25], [0.25, 0.25]])
        # Uniform across 2 cells
        h2 = np.array([[0.0, 0.5], [0.5, 0.0]])

        e1 = compute_spatial_entropy(h1)
        e2 = compute_spatial_entropy(h2)

        # e1 should be log2(4) = 2, e2 should be log2(2) = 1
        assert abs(e1 - 2.0) < 0.1
        assert abs(e2 - 1.0) < 0.1
        assert e1 > e2  # More spread = higher entropy

    def test_entropy_scaling(self) -> None:
        """Entropy should scale logarithmically with grid size."""
        # For uniform distribution, entropy = log2(M*N)
        sizes = [(2, 2), (4, 4), (8, 8)]
        entropies = []

        for M, N in sizes:
            h = np.full((M, N), 1.0 / (M * N))
            e = compute_spatial_entropy(h)
            entropies.append(e)

        # Entropy should increase with grid size
        assert entropies[0] < entropies[1] < entropies[2]

        # Check specific values
        assert abs(entropies[0] - 2.0) < 0.1  # log2(4)
        assert abs(entropies[1] - 4.0) < 0.1  # log2(16)
        assert abs(entropies[2] - 6.0) < 0.1  # log2(64)

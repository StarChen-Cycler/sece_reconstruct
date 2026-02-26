"""Unit tests for distribution function calculation.

Tests verify formulas (4), (5), (6) from the SECE paper:
- Formula (4): f_k = S_k / sum(S_l where l != k)
- Formula (5): Normalization to sum(f) = 1
- Formula (6): CDF F_k = cumulative sum of f_l
"""

from __future__ import annotations

import numpy as np
import pytest

from sece.distribution import (
    compute_distribution_function,
    validate_distribution,
)


class TestComputeDistributionFunction:
    """Tests for compute_distribution_function."""

    def test_returns_correct_shapes(self) -> None:
        """f and F should have same shape as input S."""
        S = np.array([1.0, 2.0, 3.0, 4.0])
        f, F = compute_distribution_function(S)
        assert f.shape == S.shape
        assert F.shape == S.shape

    def test_normalization_sum_to_one(self) -> None:
        """Formula (5): sum(f) should equal 1."""
        S = np.array([1.0, 2.0, 3.0])
        f, F = compute_distribution_function(S)
        assert np.isclose(np.sum(f), 1.0)

    def test_cdf_ends_at_one(self) -> None:
        """CDF F[-1] should equal 1."""
        S = np.array([1.0, 2.0, 3.0])
        f, F = compute_distribution_function(S)
        assert np.isclose(F[-1], 1.0)

    def test_cdf_monotonically_increasing(self) -> None:
        """Formula (6): F should be monotonically increasing."""
        S = np.array([0.5, 1.0, 2.0, 3.0])
        f, F = compute_distribution_function(S)
        assert np.all(np.diff(F) >= 0)

    def test_cdf_equals_cumsum(self) -> None:
        """F should equal cumsum(f)."""
        S = np.array([1.0, 2.0, 3.0])
        f, F = compute_distribution_function(S)
        np.testing.assert_allclose(F, np.cumsum(f))

    def test_single_gray_level(self) -> None:
        """K=1 edge case: single gray level maps to itself."""
        S = np.array([2.5])
        f, F = compute_distribution_function(S)
        assert f[0] == 1.0
        assert F[0] == 1.0

    def test_uniform_entropies(self) -> None:
        """When all entropies are equal, f should be uniform."""
        S = np.array([2.0, 2.0, 2.0, 2.0])
        f, F = compute_distribution_function(S)
        expected_f = np.ones(4) / 4
        np.testing.assert_allclose(f, expected_f)

    def test_zero_entropy_gray_level(self) -> None:
        """Gray level with zero entropy should have f_k near zero."""
        S = np.array([0.0, 2.0, 3.0])
        f, F = compute_distribution_function(S)
        # f[0] should be small since S[0] = 0
        # Formula (4): f_0 = S_0 / (S_1 + S_2) = 0 / 5 = 0
        assert f[0] == 0.0

    def test_higher_entropy_higher_importance(self) -> None:
        """Gray levels with higher entropy should have higher f values."""
        S = np.array([1.0, 3.0])  # S[1] > S[0]
        f, F = compute_distribution_function(S)
        # Higher entropy means more spatial dispersion, thus higher importance
        assert f[1] > f[0]

    def test_formula_4_relative_importance(self) -> None:
        """Formula (4): f_k = S_k / sum(S_l where l != k)."""
        S = np.array([1.0, 2.0, 3.0])
        f, F = compute_distribution_function(S)

        # Before normalization, check relative importance
        # f_0_raw = S_0 / (S_1 + S_2) = 1.0 / 5.0 = 0.2
        # f_1_raw = S_1 / (S_0 + S_2) = 2.0 / 4.0 = 0.5
        # f_2_raw = S_2 / (S_0 + S_1) = 3.0 / 3.0 = 1.0
        # sum_raw = 1.7
        # After normalization: f_0 = 0.2/1.7, f_1 = 0.5/1.7, f_2 = 1.0/1.7
        f_0_raw = 1.0 / (2.0 + 3.0)  # 0.2
        f_1_raw = 2.0 / (1.0 + 3.0)  # 0.5
        f_2_raw = 3.0 / (1.0 + 2.0)  # 1.0
        total_raw = f_0_raw + f_1_raw + f_2_raw  # 1.7

        expected_f = np.array([f_0_raw, f_1_raw, f_2_raw]) / total_raw
        np.testing.assert_allclose(f, expected_f)

    def test_raises_on_empty_array(self) -> None:
        """Should raise ValueError for empty S array."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_distribution_function(np.array([]))

    def test_raises_on_negative_entropy(self) -> None:
        """Should raise ValueError for negative entropy values."""
        with pytest.raises(ValueError, match="must be non-negative"):
            compute_distribution_function(np.array([1.0, -0.5, 2.0]))

    def test_with_many_gray_levels(self) -> None:
        """Test with many gray levels (256)."""
        np.random.seed(42)
        S = np.random.uniform(0.5, 3.0, 256)
        f, F = compute_distribution_function(S)

        assert np.isclose(np.sum(f), 1.0)
        assert np.isclose(F[-1], 1.0)
        assert np.all(np.diff(F) >= 0)


class TestValidateDistribution:
    """Tests for validate_distribution."""

    def test_valid_distribution(self) -> None:
        """Valid distribution should pass validation."""
        S = np.array([1.0, 2.0, 3.0])
        f, F = compute_distribution_function(S)
        assert validate_distribution(f, F)

    def test_rejects_non_normalized_f(self) -> None:
        """Should reject f that doesn't sum to 1."""
        f = np.array([0.3, 0.3, 0.3])  # Sum = 0.9, not 1.0
        F = np.cumsum(f)
        assert not validate_distribution(f, F)

    def test_rejects_wrong_cdf(self) -> None:
        """Should reject F that doesn't equal cumsum(f)."""
        f = np.array([0.2, 0.3, 0.5])
        F = np.array([0.2, 0.4, 1.0])  # Wrong: should be [0.2, 0.5, 1.0]
        assert not validate_distribution(f, F)

    def test_rejects_decreasing_cdf(self) -> None:
        """Should reject non-monotonic CDF."""
        f = np.array([0.5, 0.2, 0.3])
        F = np.array([0.5, 0.3, 0.6])  # Decreasing at index 1
        assert not validate_distribution(f, F)

    def test_rejects_cdf_not_ending_at_one(self) -> None:
        """Should reject CDF that doesn't end at 1."""
        f = np.array([0.2, 0.3, 0.4])  # Sum = 0.9
        F = np.cumsum(f)
        assert not validate_distribution(f, F)


class TestDistributionIntegration:
    """Integration tests with spatial entropy."""

    def test_distribution_from_spatial_entropy(self) -> None:
        """Test distribution computation from actual spatial entropies."""
        # Simulate spatial entropy values for a simple image
        # Image with 3 gray levels: 0, 128, 255
        S = np.array([2.5, 3.2, 1.8])  # Entropy for each level

        f, F = compute_distribution_function(S)

        # Verify properties
        assert np.isclose(np.sum(f), 1.0)
        assert np.isclose(F[-1], 1.0)
        assert np.all(np.diff(F) >= 0)

        # Level with highest entropy (128) should have highest importance
        assert f[1] > f[0]  # S[1] > S[0]
        assert f[1] > f[2]  # S[1] > S[2]

    @pytest.mark.parametrize("K", [1, 2, 10, 50, 256])
    def test_various_gray_level_counts(self, K: int) -> None:
        """Test with various numbers of gray levels."""
        np.random.seed(42)
        S = np.random.uniform(0.1, 5.0, K)
        f, F = compute_distribution_function(S)

        assert np.isclose(np.sum(f), 1.0, atol=1e-9)
        assert np.isclose(F[-1], 1.0, atol=1e-9)

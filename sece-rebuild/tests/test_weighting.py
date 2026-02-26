"""Tests for DCT coefficient weighting functions.

Tests cover:
- Alpha computation from distribution
- Weight matrix computation
- DCT coefficient weighting
- Edge cases and properties
"""

from __future__ import annotations

import numpy as np
import pytest

from sece.weighting import (
    compute_alpha,
    weight_coefficients,
    weight_coefficients_vectorized,
    compute_weight_matrix,
)


class TestComputeAlpha:
    """Tests for compute_alpha function."""

    def test_uniform_distribution(self):
        """Uniform distribution has predictable entropy."""
        # Uniform distribution over n values has entropy = log2(n)
        f = np.array([0.25, 0.25, 0.25, 0.25])  # n=4, entropy=2
        alpha = compute_alpha(f, gamma=0.5)
        expected = 2**0.5  # sqrt(2)
        np.testing.assert_allclose(alpha, expected, rtol=1e-10)

    def test_gamma_zero_returns_one(self):
        """gamma=0 returns alpha=1 (no weighting)."""
        f = np.array([0.1, 0.2, 0.3, 0.4])
        alpha = compute_alpha(f, gamma=0.0)
        np.testing.assert_allclose(alpha, 1.0, rtol=1e-10)

    def test_gamma_one_returns_entropy(self):
        """gamma=1 returns entropy directly."""
        f = np.array([0.5, 0.5])  # entropy = 1
        alpha = compute_alpha(f, gamma=1.0)
        np.testing.assert_allclose(alpha, 1.0, rtol=1e-10)

    def test_single_value_distribution(self):
        """Single non-zero value has entropy 0."""
        f = np.array([1.0, 0.0, 0.0])
        alpha = compute_alpha(f, gamma=1.0)
        np.testing.assert_allclose(alpha, 0.0, atol=1e-10)

    def test_empty_distribution_returns_one(self):
        """Empty or all-zero distribution returns 1."""
        f = np.array([0.0, 0.0, 0.0])
        alpha = compute_alpha(f, gamma=0.5)
        np.testing.assert_allclose(alpha, 1.0, rtol=1e-10)

    def test_normalization(self):
        """Non-normalized input is normalized internally."""
        f = np.array([1, 2, 3, 4])  # Sum = 10
        f_normalized = f / 10

        alpha1 = compute_alpha(f, gamma=1.0)
        alpha2 = compute_alpha(f_normalized, gamma=1.0)

        np.testing.assert_allclose(alpha1, alpha2, rtol=1e-10)

    def test_alpha_increases_with_gamma(self):
        """Alpha increases with gamma for non-uniform distribution."""
        f = np.array([0.1, 0.2, 0.3, 0.4])

        alphas = [compute_alpha(f, gamma=g) for g in [0.0, 0.25, 0.5, 0.75, 1.0]]

        # Alpha should be monotonically increasing
        for i in range(len(alphas) - 1):
            assert alphas[i] <= alphas[i + 1]

    def test_higher_entropy_higher_alpha(self):
        """Higher entropy distributions produce higher alpha (for gamma > 0)."""
        # f1 is more uniform (higher entropy)
        f1 = np.array([0.25, 0.25, 0.25, 0.25])
        # f2 is more peaked (lower entropy)
        f2 = np.array([0.7, 0.1, 0.1, 0.1])

        alpha1 = compute_alpha(f1, gamma=0.5)
        alpha2 = compute_alpha(f2, gamma=0.5)

        assert alpha1 > alpha2


class TestWeightCoefficients:
    """Tests for weight_coefficients function."""

    @pytest.fixture
    def sample_dct(self):
        """Standard 8x8 DCT coefficient matrix."""
        np.random.seed(42)
        return np.random.randn(8, 8)

    def test_output_shape(self, sample_dct):
        """Output has same shape as input."""
        result = weight_coefficients(sample_dct, alpha=2.0)
        assert result.shape == sample_dct.shape

    def test_dc_coefficient_unchanged(self, sample_dct):
        """DC coefficient (0,0) is never modified."""
        alpha = 3.0
        result = weight_coefficients(sample_dct, alpha=alpha)
        np.testing.assert_allclose(result[0, 0], sample_dct[0, 0], rtol=1e-10)

    def test_alpha_one_no_change(self):
        """alpha=1 results in no weighting."""
        D = np.random.randn(8, 8)
        result = weight_coefficients(D, alpha=1.0)
        np.testing.assert_allclose(result, D, rtol=1e-10)

    def test_higher_frequency_higher_weight(self):
        """Higher frequency coefficients get higher weights when alpha > 1."""
        D = np.ones((8, 8))  # All coefficients equal
        alpha = 2.0
        result = weight_coefficients(D, alpha=alpha)

        # (7,7) should have highest weight, (0,0) lowest
        assert result[0, 0] < result[7, 7]
        assert result[0, 0] == D[0, 0]  # DC unchanged

    def test_lower_frequency_lower_weight(self):
        """Lower frequency coefficients get lower weights when alpha > 1."""
        D = np.ones((8, 8))
        alpha = 2.0
        result = weight_coefficients(D, alpha=alpha)

        # Weight increases with frequency index
        for k in range(7):
            assert result[k, 0] <= result[k + 1, 0]
            assert result[0, k] <= result[0, k + 1]

    def test_1d_input_raises_error(self):
        """1D input raises ValueError."""
        D = np.random.randn(64)
        with pytest.raises(ValueError, match="2D"):
            weight_coefficients(D, alpha=2.0)

    def test_3d_input_raises_error(self):
        """3D input raises ValueError."""
        D = np.random.randn(8, 8, 3)
        with pytest.raises(ValueError, match="2D"):
            weight_coefficients(D, alpha=2.0)

    def test_single_element(self):
        """Single element array is returned unchanged."""
        D = np.array([[5.0]])
        result = weight_coefficients(D, alpha=2.0)
        np.testing.assert_allclose(result, D, rtol=1e-10)

    def test_rectangular_wide(self):
        """Wide rectangular array processes correctly."""
        D = np.random.randn(8, 16)
        result = weight_coefficients(D, alpha=2.0)
        assert result.shape == D.shape
        assert result[0, 0] == D[0, 0]  # DC unchanged

    def test_rectangular_tall(self):
        """Tall rectangular array processes correctly."""
        D = np.random.randn(16, 8)
        result = weight_coefficients(D, alpha=2.0)
        assert result.shape == D.shape
        assert result[0, 0] == D[0, 0]  # DC unchanged


class TestWeightCoefficientsVectorized:
    """Tests for vectorized implementation."""

    def test_matches_loop_version(self):
        """Vectorized version produces same results as loop version."""
        np.random.seed(42)
        D = np.random.randn(16, 16)
        alpha = 2.5

        result_loop = weight_coefficients(D, alpha)
        result_vec = weight_coefficients_vectorized(D, alpha)

        np.testing.assert_allclose(result_loop, result_vec, rtol=1e-12)

    def test_various_sizes(self):
        """Vectorized works for various sizes."""
        np.random.seed(42)
        alpha = 2.0

        for h, w in [(8, 8), (16, 32), (64, 64), (32, 16)]:
            D = np.random.randn(h, w)
            result_loop = weight_coefficients(D, alpha)
            result_vec = weight_coefficients_vectorized(D, alpha)
            np.testing.assert_allclose(result_loop, result_vec, rtol=1e-12)


class TestComputeWeightMatrix:
    """Tests for compute_weight_matrix function."""

    def test_dc_weight_is_one(self):
        """DC position (0,0) always has weight 1."""
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
            W = compute_weight_matrix(8, 8, alpha)
            assert W[0, 0] == 1.0

    def test_weights_increase_with_frequency(self):
        """Weights increase toward higher frequencies when alpha > 1."""
        W = compute_weight_matrix(8, 8, alpha=2.0)

        # Diagonal should increase
        for i in range(7):
            assert W[i, i] < W[i + 1, i + 1]

    def test_weights_decrease_with_alpha_less_than_one(self):
        """Weights decrease toward high frequencies when alpha < 1."""
        W = compute_weight_matrix(8, 8, alpha=0.5)

        # Diagonal should decrease
        for i in range(7):
            assert W[i, i] > W[i + 1, i + 1]

    def test_all_weights_one_for_alpha_one(self):
        """All weights are 1 when alpha=1."""
        W = compute_weight_matrix(8, 8, alpha=1.0)
        np.testing.assert_allclose(W, 1.0, rtol=1e-10)

    def test_symmetry(self):
        """Weight matrix is symmetric for square input."""
        W = compute_weight_matrix(8, 8, alpha=2.0)
        np.testing.assert_allclose(W, W.T, rtol=1e-10)

    def test_rectangular(self):
        """Works for non-square dimensions."""
        W = compute_weight_matrix(8, 16, alpha=2.0)
        assert W.shape == (8, 16)
        assert W[0, 0] == 1.0

    def test_single_element(self):
        """1x1 matrix has weight 1."""
        W = compute_weight_matrix(1, 1, alpha=2.0)
        assert W[0, 0] == 1.0


class TestWeightingProperties:
    """Tests for mathematical properties of weighting."""

    def test_linearity(self):
        """Weighting is linear: w(a*D1 + b*D2) = a*w(D1) + b*w(D2)."""
        np.random.seed(42)
        D1 = np.random.randn(8, 8)
        D2 = np.random.randn(8, 8)
        a, b = 2.0, 3.0
        alpha = 2.0

        W_combined = weight_coefficients(a * D1 + b * D2, alpha)
        W_separate = a * weight_coefficients(D1, alpha) + b * weight_coefficients(D2, alpha)

        np.testing.assert_allclose(W_combined, W_separate, rtol=1e-10)

    def test_separability(self):
        """2D weight is separable: w(k,l) = w_k(k) * w_l(l)."""
        H, W = 8, 8
        alpha = 2.0

        # Full weight matrix
        weights_2d = compute_weight_matrix(H, W, alpha)

        # Separable 1D weights
        k_indices = np.arange(H)
        l_indices = np.arange(W)
        w_k = 1.0 + (alpha - 1) * k_indices / (H - 1)
        w_l = 1.0 + (alpha - 1) * l_indices / (W - 1)

        # Outer product should match 2D weights
        weights_separable = np.outer(w_k, w_l)

        np.testing.assert_allclose(weights_2d, weights_separable, rtol=1e-10)

    def test_preserves_sign(self):
        """Weighting preserves sign of coefficients."""
        np.random.seed(42)
        D = np.random.randn(8, 8)
        alpha = 3.0
        result = weight_coefficients(D, alpha)

        # Sign should be preserved (weights are always positive)
        assert np.all(np.sign(result) == np.sign(D))


class TestEdgeCases:
    """Edge case tests."""

    def test_large_alpha(self):
        """Large alpha values don't cause overflow."""
        D = np.random.randn(8, 8)
        result = weight_coefficients(D, alpha=100.0)
        assert np.all(np.isfinite(result))

    def test_small_alpha(self):
        """Small alpha values work correctly."""
        D = np.random.randn(8, 8)
        result = weight_coefficients(D, alpha=0.01)
        assert np.all(np.isfinite(result))

    def test_zeros(self):
        """Zero DCT coefficients remain zero after weighting."""
        D = np.zeros((8, 8))
        result = weight_coefficients(D, alpha=2.0)
        np.testing.assert_allclose(result, 0, atol=1e-15)

    def test_constant_dct(self):
        """Constant DCT (only DC) is unchanged."""
        D = np.zeros((8, 8))
        D[0, 0] = 100.0
        result = weight_coefficients(D, alpha=2.0)
        np.testing.assert_allclose(result, D, rtol=1e-10)

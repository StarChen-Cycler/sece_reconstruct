"""Tests for 2D-DCT transform functions.

Tests cover:
- Forward and inverse DCT accuracy
- Round-trip error bounds
- Energy preservation (Parseval's theorem)
- Edge cases (small, large, non-square images)
"""

from __future__ import annotations

import numpy as np
import pytest

from sece.dct import dct2d, idct2d, dct2d_blockwise, idct2d_blockwise


class TestDCT2D:
    """Tests for dct2d function."""

    @pytest.fixture
    def sample_2d(self):
        """Standard 64x64 test array."""
        np.random.seed(42)
        return np.random.randn(64, 64)

    def test_output_shape(self, sample_2d):
        """Output has same shape as input."""
        result = dct2d(sample_2d)
        assert result.shape == sample_2d.shape

    def test_output_dtype(self, sample_2d):
        """Output is float64."""
        result = dct2d(sample_2d)
        assert result.dtype == np.float64

    def test_dc_coefficient(self, sample_2d):
        """DC coefficient relates to mean value."""
        result = dct2d(sample_2d)
        # For orthonormal DCT, DC = sum(x) / sqrt(M*N) = mean * sqrt(M*N)
        H, W = sample_2d.shape
        expected_dc = np.mean(sample_2d) * np.sqrt(H * W)
        np.testing.assert_allclose(result[0, 0], expected_dc, rtol=1e-10)

    def test_energy_preservation(self, sample_2d):
        """Orthonormal DCT preserves energy (Parseval's theorem)."""
        result = dct2d(sample_2d)
        input_energy = np.sum(sample_2d**2)
        output_energy = np.sum(result**2)
        np.testing.assert_allclose(input_energy, output_energy, rtol=1e-10)

    def test_1d_input_raises_error(self):
        """1D input raises ValueError."""
        x = np.random.randn(64)
        with pytest.raises(ValueError, match="2D"):
            dct2d(x)

    def test_3d_input_raises_error(self):
        """3D input raises ValueError."""
        x = np.random.randn(8, 8, 3)
        with pytest.raises(ValueError, match="2D"):
            dct2d(x)

    def test_list_input_converted(self):
        """List input is converted to numpy array."""
        x = [[1, 2], [3, 4]]
        result = dct2d(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)


class TestIDCT2D:
    """Tests for idct2d function."""

    @pytest.fixture
    def sample_2d(self):
        """Standard 64x64 test array."""
        np.random.seed(42)
        return np.random.randn(64, 64)

    def test_output_shape(self, sample_2d):
        """Output has same shape as input."""
        result = idct2d(sample_2d)
        assert result.shape == sample_2d.shape

    def test_output_dtype(self, sample_2d):
        """Output is float64."""
        result = idct2d(sample_2d)
        assert result.dtype == np.float64

    def test_1d_input_raises_error(self):
        """1D input raises ValueError."""
        D = np.random.randn(64)
        with pytest.raises(ValueError, match="2D"):
            idct2d(D)


class TestRoundTrip:
    """Tests for DCT round-trip accuracy."""

    def test_roundtrip_small(self):
        """Round-trip MSE < 1e-10 for small image."""
        np.random.seed(42)
        x = np.random.randn(32, 32)
        D = dct2d(x)
        recovered = idct2d(D)
        mse = np.mean((x - recovered) ** 2)
        assert mse < 1e-10, f"Round-trip MSE too high: {mse}"

    def test_roundtrip_100x100(self):
        """Round-trip MSE < 1e-10 for 100x100 image (success criterion)."""
        np.random.seed(42)
        x = np.random.randn(100, 100)
        D = dct2d(x)
        recovered = idct2d(D)
        mse = np.mean((x - recovered) ** 2)
        assert mse < 1e-10, f"Round-trip MSE too high: {mse}"

    def test_roundtrip_medium(self):
        """Round-trip MSE < 1e-10 for medium image."""
        np.random.seed(42)
        x = np.random.randn(256, 256)
        D = dct2d(x)
        recovered = idct2d(D)
        mse = np.mean((x - recovered) ** 2)
        assert mse < 1e-10, f"Round-trip MSE too high: {mse}"

    def test_roundtrip_large(self):
        """Round-trip MSE < 1e-10 for large image."""
        np.random.seed(42)
        x = np.random.randn(512, 512)
        D = dct2d(x)
        recovered = idct2d(D)
        mse = np.mean((x - recovered) ** 2)
        assert mse < 1e-10, f"Round-trip MSE too high: {mse}"

    def test_roundtrip_exact(self):
        """Round-trip is nearly exact."""
        np.random.seed(42)
        x = np.random.randn(64, 64)
        recovered = idct2d(dct2d(x))
        np.testing.assert_allclose(x, recovered, rtol=1e-10, atol=1e-14)


class TestNonSquareImages:
    """Tests for non-square (rectangular) images."""

    def test_wide_image(self):
        """Wide image (width > height) processes correctly."""
        np.random.seed(42)
        x = np.random.randn(64, 128)
        D = dct2d(x)
        recovered = idct2d(D)
        np.testing.assert_allclose(x, recovered, rtol=1e-10, atol=1e-14)

    def test_tall_image(self):
        """Tall image (height > width) processes correctly."""
        np.random.seed(42)
        x = np.random.randn(128, 64)
        D = dct2d(x)
        recovered = idct2d(D)
        np.testing.assert_allclose(x, recovered, rtol=1e-10, atol=1e-14)

    def test_very_wide_image(self):
        """Very wide image processes correctly."""
        np.random.seed(42)
        x = np.random.randn(32, 256)
        D = dct2d(x)
        recovered = idct2d(D)
        np.testing.assert_allclose(x, recovered, rtol=1e-10, atol=1e-14)


class TestEdgeCases:
    """Edge case tests."""

    def test_2x2_image(self):
        """2x2 image processes correctly."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        D = dct2d(x)
        recovered = idct2d(D)
        np.testing.assert_allclose(x, recovered, rtol=1e-10, atol=1e-14)

    def test_1x1_image(self):
        """1x1 image processes correctly."""
        x = np.array([[5.0]])
        D = dct2d(x)
        recovered = idct2d(D)
        np.testing.assert_allclose(x, recovered, rtol=1e-12)

    def test_zeros(self):
        """Zero array returns zeros."""
        x = np.zeros((32, 32))
        D = dct2d(x)
        np.testing.assert_allclose(D, 0, atol=1e-15)

    def test_constant_image(self):
        """Constant image has only DC coefficient."""
        x = np.full((32, 32), 10.0)
        D = dct2d(x)
        # DC coefficient should be non-zero
        assert D[0, 0] != 0
        # All AC coefficients should be essentially zero
        ac = D[1:, :].flatten()
        np.testing.assert_allclose(ac, 0, atol=1e-12)
        ac = D[:, 1:].flatten()
        np.testing.assert_allclose(ac, 0, atol=1e-12)

    def test_symmetric_input(self):
        """Symmetric input produces expected DCT pattern."""
        # Create a simple symmetric pattern
        x = np.array(
            [
                [1, 2, 2, 1],
                [2, 3, 3, 2],
                [2, 3, 3, 2],
                [1, 2, 2, 1],
            ],
            dtype=np.float64,
        )
        D = dct2d(x)
        recovered = idct2d(D)
        np.testing.assert_allclose(x, recovered, rtol=1e-10, atol=1e-14)


class TestBlockwiseDCT:
    """Tests for blockwise DCT functions."""

    @pytest.fixture
    def sample_image(self):
        """Standard 64x64 test image."""
        np.random.seed(42)
        return np.random.randn(64, 64)

    def test_blockwise_output_shape(self, sample_image):
        """Blockwise DCT output has same shape."""
        result = dct2d_blockwise(sample_image, block_size=8)
        assert result.shape == sample_image.shape

    def test_blockwise_roundtrip(self, sample_image):
        """Blockwise round-trip recovers original."""
        D = dct2d_blockwise(sample_image, block_size=8)
        recovered = idct2d_blockwise(D, block_size=8)
        np.testing.assert_allclose(sample_image, recovered, rtol=1e-10, atol=1e-14)

    def test_blockwise_non_divisible_size(self):
        """Blockwise DCT handles non-block-divisible sizes with padding."""
        np.random.seed(42)
        x = np.random.randn(65, 65)  # Not divisible by 8
        D = dct2d_blockwise(x, block_size=8)
        recovered = idct2d_blockwise(D, block_size=8)
        # Edge artifacts from padding are expected; check that center is accurate
        # Only verify the area that doesn't have edge effects
        center = recovered[:64, :64]
        original_center = x[:64, :64]
        mse = np.mean((original_center - center) ** 2)
        assert mse < 1e-10, f"Blockwise center MSE too high: {mse}"

    def test_blockwise_different_sizes(self):
        """Blockwise DCT with different block sizes."""
        np.random.seed(42)
        x = np.random.randn(64, 64)

        for block_size in [2, 4, 8, 16]:
            D = dct2d_blockwise(x, block_size=block_size)
            recovered = idct2d_blockwise(D, block_size=block_size)
            np.testing.assert_allclose(x, recovered, rtol=1e-10, atol=1e-14)

    def test_blockwise_vs_full_different(self, sample_image):
        """Blockwise DCT differs from full DCT."""
        D_full = dct2d(sample_image)
        D_block = dct2d_blockwise(sample_image, block_size=8)
        # They should be different
        assert not np.allclose(D_full, D_block)


class TestDCTProperties:
    """Tests for mathematical properties of DCT."""

    def test_linearity(self):
        """DCT is linear: dct2d(a*x + b*y) = a*dct2d(x) + b*dct2d(y)."""
        np.random.seed(42)
        x = np.random.randn(32, 32)
        y = np.random.randn(32, 32)
        a, b = 2.0, 3.0

        D_combined = dct2d(a * x + b * y)
        D_separate = a * dct2d(x) + b * dct2d(y)

        np.testing.assert_allclose(D_combined, D_separate, rtol=1e-12)

    def test_separability(self):
        """2D-DCT is separable: DCT2D = DCT_row(DCT_col(x))."""
        from scipy.fftpack import dct

        np.random.seed(42)
        x = np.random.randn(32, 32)

        # Direct 2D-DCT
        D_direct = dct2d(x)

        # Separable 1D-DCTs (column-wise then row-wise)
        D_separable = dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")

        np.testing.assert_allclose(D_direct, D_separable, rtol=1e-10, atol=1e-14)

    def test_orthonormality(self):
        """Orthonormal DCT preserves inner products."""
        np.random.seed(42)
        x = np.random.randn(32, 32)
        y = np.random.randn(32, 32)

        # Inner product in spatial domain
        spatial_inner = np.sum(x * y)

        # Inner product in DCT domain
        X = dct2d(x)
        Y = dct2d(y)
        dct_inner = np.sum(X * Y)

        np.testing.assert_allclose(spatial_inner, dct_inner, rtol=1e-10)

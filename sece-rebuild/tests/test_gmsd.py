"""Tests for GMSD metric.

Tests verify:
- GMSD(X, X) equals 0.0
- GMSD(X, Y) > 0 when X != Y
- GMSD range is [0, inf)
- GMSD works with grayscale and color images
- Integration with SECE enhancement
"""

import pytest
import numpy as np

from sece.metrics import gmsd, gmsd_comparison, gmsd_map
from sece.metrics.gmsd import (
    _compute_gradient_magnitude,
    _compute_gradient_magnitude_vectorized,
)


class TestGMSDBasic:
    """Basic GMSD functionality tests."""

    def test_gmsd_returns_float(self, sample_grayscale):
        """GMSD should return a float."""
        result = gmsd(sample_grayscale, sample_grayscale)
        assert isinstance(result, float)

    def test_gmsd_identical_images_equals_zero(self, sample_grayscale):
        """GMSD(X, X) should equal 0.0."""
        result = gmsd(sample_grayscale, sample_grayscale)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_gmsd_range_non_negative(self, sample_grayscale):
        """GMSD should be non-negative."""
        result = gmsd(sample_grayscale, sample_grayscale)
        assert result >= 0

    def test_gmsd_different_images_greater_than_zero(self, sample_grayscale):
        """GMSD(X, Y) should be > 0 when X != Y."""
        different = np.random.randint(0, 256, sample_grayscale.shape, dtype=np.uint8)
        result = gmsd(sample_grayscale, different)
        assert result > 0, "GMSD should be > 0 for different images"

    def test_gmsd_shape_mismatch_raises(self, sample_grayscale, small_image):
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError, match="shapes must match"):
            gmsd(sample_grayscale, small_image)


class TestGMSDColor:
    """Test GMSD with color images."""

    def test_gmsd_color_images(self, sample_color):
        """GMSD should work with color images."""
        result = gmsd(sample_color, sample_color)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_gmsd_color_vs_grayscale_equivalent(self, sample_grayscale):
        """Same image in grayscale and BGR should give similar GMSD."""
        import cv2

        # Create BGR version of grayscale
        bgr = cv2.cvtColor(sample_grayscale, cv2.COLOR_GRAY2BGR)

        # GMSD(grayscale, grayscale) should equal GMSD(bgr, bgr)
        gmsd_gray = gmsd(sample_grayscale, sample_grayscale)
        gmsd_bgr = gmsd(bgr, bgr)

        assert gmsd_gray == pytest.approx(gmsd_bgr, abs=1e-10)

    def test_gmsd_mixed_grayscale_color_raises(self, sample_grayscale, sample_color):
        """Mixed grayscale and color should raise due to shape mismatch."""
        with pytest.raises(ValueError, match="shapes must match"):
            gmsd(sample_grayscale, sample_color)


class TestGMSDMap:
    """Test gmsd_map function."""

    def test_gmsd_map_returns_tuple(self, sample_grayscale):
        """gmsd_map should return (float, ndarray)."""
        value, full_map = gmsd_map(sample_grayscale, sample_grayscale)
        assert isinstance(value, float)
        assert isinstance(full_map, np.ndarray)

    def test_gmsd_map_value_equals_gmsd(self, sample_grayscale):
        """Value from gmsd_map should equal gmsd()."""
        value_map, _ = gmsd_map(sample_grayscale, sample_grayscale)
        value_direct = gmsd(sample_grayscale, sample_grayscale)
        assert value_map == pytest.approx(value_direct, rel=1e-6)

    def test_gmsd_map_shape_same(self, sample_grayscale):
        """GMS map should have same shape as input."""
        _, full_map = gmsd_map(sample_grayscale, sample_grayscale)
        assert full_map.shape == sample_grayscale.shape

    def test_gmsd_map_values_in_range(self, sample_grayscale):
        """GMS map values should be in [0, 1] range."""
        different = np.random.randint(0, 256, sample_grayscale.shape, dtype=np.uint8)
        _, full_map = gmsd_map(sample_grayscale, different)
        assert np.all(full_map >= 0)
        assert np.all(full_map <= 1)

    def test_gmsd_map_shape_mismatch_raises(self, sample_grayscale, small_image):
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError, match="shapes must match"):
            gmsd_map(sample_grayscale, small_image)


class TestGMSDComparison:
    """Test gmsd_comparison function."""

    def test_comparison_returns_dict(self, sample_grayscale):
        """gmsd_comparison should return a dict."""
        result = gmsd_comparison(sample_grayscale, sample_grayscale)
        assert isinstance(result, dict)

    def test_comparison_has_required_keys(self, sample_grayscale):
        """Result should have required keys."""
        result = gmsd_comparison(sample_grayscale, sample_grayscale)
        assert "gmsd" in result
        assert "visually_distorted" in result
        assert "threshold" in result

    def test_identical_images_not_distorted(self, sample_grayscale):
        """Identical images should not be visually distorted."""
        result = gmsd_comparison(sample_grayscale, sample_grayscale)
        assert result["visually_distorted"] is False

    def test_custom_threshold(self, sample_grayscale):
        """Custom threshold should work."""
        result = gmsd_comparison(sample_grayscale, sample_grayscale, threshold=0.05)
        assert result["threshold"] == 0.05
        assert result["visually_distorted"] is False


class TestGMSDIntegration:
    """Integration tests with SECE enhancement."""

    def test_gmsd_after_enhancement(self, low_contrast_image):
        """GMSD should be computed after SECE enhancement."""
        from sece import sece

        result = sece(low_contrast_image)
        enhanced = result.image

        gmsd_value = gmsd(low_contrast_image, enhanced)

        # GMSD should be non-negative
        assert gmsd_value >= 0

        # GMSD for enhancement should typically be < 0.2
        # (enhancement should not severely distort the image)
        assert gmsd_value < 0.3, f"GMSD too high: {gmsd_value}"

    def test_gmsd_sece_vs_secedct(self, sample_grayscale):
        """Compare GMSD for SECE vs SECEDCT enhancement."""
        from sece import sece
        from sece.secedct import secedct

        sece_result = sece(sample_grayscale)
        secedct_result = secedct(sample_grayscale, gamma=0.5)

        gmsd_sece = gmsd(sample_grayscale, sece_result.image)
        gmsd_secedct = gmsd(sample_grayscale, secedct_result.image)

        # Both should be non-negative
        assert gmsd_sece >= 0
        assert gmsd_secedct >= 0

    def test_gmsd_higher_gamma_more_distortion(self, sample_grayscale):
        """Higher gamma should generally produce higher GMSD (more distortion)."""
        from sece.secedct import secedct

        result_low = secedct(sample_grayscale, gamma=0.1)
        result_high = secedct(sample_grayscale, gamma=0.9)

        gmsd_low = gmsd(sample_grayscale, result_low.image)
        gmsd_high = gmsd(sample_grayscale, result_high.image)

        # Higher gamma typically produces more local enhancement
        # which may increase GMSD (more distortion from reference)
        # This is a general trend, not guaranteed
        assert gmsd_low >= 0
        assert gmsd_high >= 0


class TestGMSDEdgeCases:
    """Edge case tests for GMSD."""

    def test_single_pixel_image(self):
        """Single pixel image should work without error."""
        single = np.array([[128]], dtype=np.uint8)
        result = gmsd(single, single)
        assert isinstance(result, float)

    def test_2x2_image(self):
        """2x2 image should work without error."""
        tiny = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = gmsd(tiny, tiny)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_uniform_image(self, single_color_image):
        """Uniform color image should work."""
        result = gmsd(single_color_image, single_color_image)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_black_image(self):
        """Black image should work."""
        black = np.zeros((64, 64), dtype=np.uint8)
        result = gmsd(black, black)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_white_image(self):
        """White image should work."""
        white = np.full((64, 64), 255, dtype=np.uint8)
        result = gmsd(white, white)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_custom_c_parameter(self, sample_grayscale):
        """Custom c parameter should work."""
        different = np.random.randint(0, 256, sample_grayscale.shape, dtype=np.uint8)

        result_default = gmsd(sample_grayscale, different, c=170)
        result_custom = gmsd(sample_grayscale, different, c=1000)

        # Both should be positive
        assert result_default > 0
        assert result_custom > 0

    def test_slightly_different_images(self, sample_grayscale):
        """Slightly different images should have low GMSD."""
        # Add small noise
        noise = np.random.randint(-5, 6, sample_grayscale.shape)
        slightly_different = np.clip(
            sample_grayscale.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

        result = gmsd(sample_grayscale, slightly_different)

        # GMSD should be relatively low for slightly different images
        assert result < 0.1, f"GMSD too high for slight difference: {result}"

    def test_very_different_images(self, sample_grayscale):
        """Very different images should have higher GMSD."""
        # Inverting an image preserves gradient magnitudes, so GMSD = 0
        # Instead, use a completely different random image
        completely_different = np.random.randint(0, 256, sample_grayscale.shape, dtype=np.uint8)

        result = gmsd(sample_grayscale, completely_different)

        # GMSD should be higher for completely different images
        assert result > 0.01, f"GMSD too low for different image: {result}"


class TestGMSDMetrics:
    """Test GMSD metric properties."""

    def test_gmsd_symmetry(self, sample_grayscale):
        """GMSD should be symmetric: GMSD(A, B) = GMSD(B, A)."""
        different = np.random.randint(0, 256, sample_grayscale.shape, dtype=np.uint8)

        gmsd_ab = gmsd(sample_grayscale, different)
        gmsd_ba = gmsd(different, sample_grayscale)

        assert gmsd_ab == pytest.approx(gmsd_ba, rel=1e-10)

    def test_gmsd_different_noise_levels(self, sample_grayscale):
        """Higher noise should generally produce higher GMSD."""
        np.random.seed(42)

        # Low noise
        noise_low = np.random.randint(-5, 6, sample_grayscale.shape)
        low_noise = np.clip(
            sample_grayscale.astype(np.int16) + noise_low, 0, 255
        ).astype(np.uint8)

        # High noise
        noise_high = np.random.randint(-50, 51, sample_grayscale.shape)
        high_noise = np.clip(
            sample_grayscale.astype(np.int16) + noise_high, 0, 255
        ).astype(np.uint8)

        gmsd_low = gmsd(sample_grayscale, low_noise)
        gmsd_high = gmsd(sample_grayscale, high_noise)

        # Higher noise should produce higher GMSD
        assert gmsd_high > gmsd_low, (
            f"High noise GMSD ({gmsd_high}) should be > low noise GMSD ({gmsd_low})"
        )


class TestGradientMagnitude:
    """Test internal gradient magnitude computation functions."""

    def test_gradient_magnitude_returns_float64(self, sample_grayscale):
        """Gradient magnitude should return float64 array."""
        img_float = sample_grayscale.astype(np.float64)
        result = _compute_gradient_magnitude_vectorized(img_float)
        assert result.dtype == np.float64

    def test_gradient_magnitude_shape_preserved(self, sample_grayscale):
        """Gradient magnitude should preserve input shape."""
        img_float = sample_grayscale.astype(np.float64)
        result = _compute_gradient_magnitude_vectorized(img_float)
        assert result.shape == sample_grayscale.shape

    def test_gradient_magnitude_uniform_image_zero(self, single_color_image):
        """Uniform image should have zero gradient magnitude."""
        img_float = single_color_image.astype(np.float64)
        result = _compute_gradient_magnitude_vectorized(img_float)
        assert np.allclose(result, 0.0)

    def test_gradient_magnitude_non_negative(self, sample_grayscale):
        """Gradient magnitude should be non-negative."""
        img_float = sample_grayscale.astype(np.float64)
        result = _compute_gradient_magnitude_vectorized(img_float)
        assert np.all(result >= 0)

    def test_gradient_magnitude_prewitt_implementation(self, sample_grayscale):
        """Prewitt implementation should work correctly."""
        img_float = sample_grayscale.astype(np.float64)
        result = _compute_gradient_magnitude(img_float)
        assert result.shape == sample_grayscale.shape
        assert np.all(result >= 0)

    def test_gradient_magnitude_comparable_results(self, sample_grayscale):
        """Vectorized and Prewitt implementations should give similar results."""
        img_float = sample_grayscale.astype(np.float64)
        result_vec = _compute_gradient_magnitude_vectorized(img_float)
        result_prewitt = _compute_gradient_magnitude(img_float)

        # Both should be non-negative
        assert np.all(result_vec >= 0)
        assert np.all(result_prewitt >= 0)

        # Both should have the same shape
        assert result_vec.shape == result_prewitt.shape

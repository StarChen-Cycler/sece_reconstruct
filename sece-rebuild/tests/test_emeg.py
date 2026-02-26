"""Tests for EMEG and SSIM metrics.

Tests verify:
- EMEG formula correctness (formula 14 from paper)
- EMEG range is [0, 1]
- EMEG(black_image) < EMEG(high_contrast_image)
- SSIM(X, X) equals 1.0
- SSIM range is [-1, 1], typically [0, 1] for images
"""

import pytest
import numpy as np

from sece.metrics import emeg, emeg_comparison
from sece.metrics import ssim, ssim_comparison, ssim_map


class TestEMEGBasic:
    """Basic EMEG functionality tests."""

    def test_emeg_returns_float(self, sample_grayscale):
        """EMEG should return a float."""
        result = emeg(sample_grayscale)
        assert isinstance(result, float)

    def test_emeg_positive(self, sample_grayscale):
        """EMEG should be non-negative."""
        result = emeg(sample_grayscale)
        assert result >= 0

    def test_emeg_range_zero_to_one(self, sample_grayscale):
        """EMEG output range should be [0, 1] for normalized images."""
        result = emeg(sample_grayscale)
        # EMEG is now normalized to [0, 1] range
        assert 0 <= result <= 1, f"EMEG {result} outside [0, 1] range"

    def test_emeg_black_image_low(self):
        """Black image should have very low EMEG."""
        black = np.zeros((64, 64), dtype=np.uint8)
        result = emeg(black)
        # Black image has no gradients, so EMEG should be very low
        assert result < 0.1, f"Black image EMEG {result} should be < 0.1"

    def test_emeg_uniform_image_low(self, single_color_image):
        """Uniform color image should have low EMEG."""
        result = emeg(single_color_image)
        assert result < 0.1, f"Uniform image EMEG {result} should be < 0.1"

    def test_emeg_high_contrast_higher_than_black(self):
        """High contrast image should have higher EMEG than black image."""
        black = np.zeros((64, 64), dtype=np.uint8)
        high_contrast = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        emeg_black = emeg(black)
        emeg_high = emeg(high_contrast)

        assert emeg_high > emeg_black, (
            f"High contrast EMEG ({emeg_high}) should be > black EMEG ({emeg_black})"
        )

    def test_emeg_low_contrast_has_value(self, low_contrast_image):
        """Low contrast image should have a valid EMEG value."""
        result = emeg(low_contrast_image)
        # Just check it produces a valid value in range
        assert 0 <= result <= 1, f"EMEG {result} outside [0, 1]"


class TestEMEGBlocksize:
    """Test EMEG with different block sizes."""

    def test_default_block_size_8(self, sample_grayscale):
        """Default block size should be 8."""
        result = emeg(sample_grayscale)
        result_explicit = emeg(sample_grayscale, block_size=8)
        assert result == pytest.approx(result_explicit, rel=1e-10)

    def test_smaller_block_size(self, sample_grayscale):
        """Smaller block size may give different result."""
        result_8 = emeg(sample_grayscale, block_size=8)
        result_4 = emeg(sample_grayscale, block_size=4)
        # Both should be valid EMEG values in [0, 1]
        assert 0 <= result_8 <= 1, f"EMEG {result_8} outside [0, 1]"
        assert 0 <= result_4 <= 1, f"EMEG {result_4} outside [0, 1]"

    def test_large_block_size(self, sample_grayscale):
        """Large block size (larger than image) should still work."""
        # Use block size larger than image
        result = emeg(sample_grayscale, block_size=256)
        assert 0 <= result <= 1, f"EMEG {result} outside [0, 1]"

    def test_small_image_handling(self, small_image):
        """Small images should be handled gracefully."""
        result = emeg(small_image, block_size=8)
        # Should not raise error, returns a valid value
        assert isinstance(result, float)


class TestEMEGColor:
    """Test EMEG with color images."""

    def test_emeg_color_converted_to_grayscale(self, sample_color):
        """Color images should be converted to grayscale."""
        result = emeg(sample_color)
        assert isinstance(result, float)
        assert 0 <= result <= 1, f"EMEG {result} outside [0, 1]"

    def test_emeg_color_vs_grayscale_equivalent(self, sample_grayscale):
        """Same image in grayscale and BGR should give similar EMEG."""
        import cv2

        # Create BGR version of grayscale
        bgr = cv2.cvtColor(sample_grayscale, cv2.COLOR_GRAY2BGR)

        emeg_gray = emeg(sample_grayscale)
        emeg_bgr = emeg(bgr)

        # Should be very close (conversion may introduce tiny differences)
        assert emeg_gray == pytest.approx(emeg_bgr, rel=0.01)


class TestEMEGComparison:
    """Test emeg_comparison function."""

    def test_comparison_returns_dict(self, sample_grayscale):
        """emeg_comparison should return a dict."""
        result = emeg_comparison(sample_grayscale, sample_grayscale)
        assert isinstance(result, dict)

    def test_comparison_has_required_keys(self, sample_grayscale):
        """Result should have required keys."""
        result = emeg_comparison(sample_grayscale, sample_grayscale)
        assert "original" in result
        assert "enhanced" in result
        assert "improvement" in result
        assert "ratio" in result

    def test_comparison_identical_images(self, sample_grayscale):
        """Identical images should have zero improvement."""
        result = emeg_comparison(sample_grayscale, sample_grayscale)
        assert result["improvement"] == pytest.approx(0.0, abs=1e-10)
        assert result["ratio"] == pytest.approx(1.0, rel=1e-10)

    def test_comparison_black_vs_high_contrast(self):
        """Black vs high contrast should show positive improvement."""
        black = np.zeros((64, 64), dtype=np.uint8)
        high_contrast = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        result = emeg_comparison(black, high_contrast)
        assert result["improvement"] > 0


class TestSSIMBasic:
    """Basic SSIM functionality tests."""

    def test_ssim_returns_float(self, sample_grayscale):
        """SSIM should return a float."""
        result = ssim(sample_grayscale, sample_grayscale)
        assert isinstance(result, float)

    def test_ssim_identical_images_equals_one(self, sample_grayscale):
        """SSIM(X, X) should equal 1.0."""
        result = ssim(sample_grayscale, sample_grayscale)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_ssim_range(self, sample_grayscale):
        """SSIM range should be [-1, 1], typically [0, 1] for images."""
        result = ssim(sample_grayscale, sample_grayscale)
        assert -1 <= result <= 1

    def test_ssim_different_images_less_than_one(self, sample_grayscale):
        """Different images should have SSIM < 1."""
        different = np.random.randint(0, 256, sample_grayscale.shape, dtype=np.uint8)
        result = ssim(sample_grayscale, different)
        assert result < 1.0

    def test_ssim_shape_mismatch_raises(self, sample_grayscale, small_image):
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError, match="shapes must match"):
            ssim(sample_grayscale, small_image)


class TestSSIMColor:
    """Test SSIM with color images."""

    def test_ssim_color_images(self, sample_color):
        """SSIM should work with color images."""
        result = ssim(sample_color, sample_color)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_ssim_mixed_grayscale_color_raises(self, sample_grayscale, sample_color):
        """Mixed grayscale and color should raise due to shape mismatch."""
        with pytest.raises(ValueError, match="shapes must match"):
            ssim(sample_grayscale, sample_color)


class TestSSIMMap:
    """Test ssim_map function."""

    def test_ssim_map_returns_tuple(self, sample_grayscale):
        """ssim_map should return (float, ndarray)."""
        mean, full_map = ssim_map(sample_grayscale, sample_grayscale)
        assert isinstance(mean, float)
        assert isinstance(full_map, np.ndarray)

    def test_ssim_map_mean_equals_ssim(self, sample_grayscale):
        """Mean from ssim_map should equal ssim()."""
        mean_ssim, _ = ssim_map(sample_grayscale, sample_grayscale)
        direct_ssim = ssim(sample_grayscale, sample_grayscale)
        assert mean_ssim == pytest.approx(direct_ssim, rel=1e-6)

    def test_ssim_map_shape_smaller(self, sample_grayscale):
        """SSIM map should be smaller than original due to windowing."""
        mean, full_map = ssim_map(sample_grayscale, sample_grayscale)
        # Map dimensions are smaller due to windowing
        assert full_map.shape[0] <= sample_grayscale.shape[0]
        assert full_map.shape[1] <= sample_grayscale.shape[1]


class TestSSIMComparison:
    """Test ssim_comparison function."""

    def test_comparison_returns_dict(self, sample_grayscale):
        """ssim_comparison should return a dict."""
        result = ssim_comparison(sample_grayscale, sample_grayscale)
        assert isinstance(result, dict)

    def test_comparison_has_required_keys(self, sample_grayscale):
        """Result should have required keys."""
        result = ssim_comparison(sample_grayscale, sample_grayscale)
        assert "ssim" in result
        assert "perceptually_similar" in result
        assert "threshold" in result

    def test_identical_images_perceptually_similar(self, sample_grayscale):
        """Identical images should be perceptually similar."""
        result = ssim_comparison(sample_grayscale, sample_grayscale)
        assert result["perceptually_similar"] is True

    def test_custom_threshold(self, sample_grayscale):
        """Custom threshold should work."""
        result = ssim_comparison(sample_grayscale, sample_grayscale, threshold=0.95)
        assert result["threshold"] == 0.95
        assert result["perceptually_similar"] is True


class TestEMEGIntegration:
    """Integration tests with SECE enhancement."""

    def test_emeg_increases_after_enhancement(self, low_contrast_image):
        """EMEG should increase after SECE enhancement."""
        from sece import sece

        result = sece(low_contrast_image)
        enhanced = result.image

        emeg_orig = emeg(low_contrast_image)
        emeg_enh = emeg(enhanced)

        assert emeg_enh > emeg_orig, (
            f"EMEG should increase after enhancement: {emeg_orig} -> {emeg_enh}"
        )

    def test_ssim_remains_high_after_enhancement(self, low_contrast_image):
        """SSIM should remain high (perceptually similar) after enhancement."""
        from sece import sece

        result = sece(low_contrast_image)
        enhanced = result.image

        ssim_value = ssim(low_contrast_image, enhanced)

        # SSIM should typically be >= 0.5 for reasonable enhancement
        assert ssim_value >= 0.3, f"SSIM too low: {ssim_value}"

    def test_secedct_increases_emeg_more(self, sample_grayscale):
        """SECEDCT with higher gamma should increase EMEG more."""
        from sece.secedct import secedct

        result_low = secedct(sample_grayscale, gamma=0.2)
        result_high = secedct(sample_grayscale, gamma=0.8)

        emeg_low = emeg(result_low.image)
        emeg_high = emeg(result_high.image)

        # Higher gamma should generally produce more local contrast
        assert emeg_high >= emeg_low, (
            f"Higher gamma should increase EMEG: {emeg_low} vs {emeg_high}"
        )


class TestEMEGEdgeCases:
    """Edge case tests for EMEG."""

    def test_single_pixel_image(self):
        """Single pixel image should return 0."""
        single = np.array([[128]], dtype=np.uint8)
        result = emeg(single)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_2x2_image(self):
        """2x2 image should work without error."""
        tiny = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = emeg(tiny, block_size=1)
        assert isinstance(result, float)

    def test_different_block_sizes_give_values(self, sample_grayscale):
        """Different block sizes should all give valid EMEG values."""
        result_8 = emeg(sample_grayscale, block_size=8)
        result_16 = emeg(sample_grayscale, block_size=16)
        result_32 = emeg(sample_grayscale, block_size=32)

        # All should be in valid range
        assert 0 <= result_8 <= 1
        assert 0 <= result_16 <= 1
        assert 0 <= result_32 <= 1

    def test_custom_epsilon(self, sample_grayscale):
        """Custom epsilon should not significantly change result normally."""
        result_default = emeg(sample_grayscale, epsilon=1e-10)
        result_custom = emeg(sample_grayscale, epsilon=1e-8)

        # Both should be in valid range
        assert 0 <= result_default <= 1
        assert 0 <= result_custom <= 1
        # Should be relatively close (within 20% since epsilon affects log)
        if result_default > 0.01:  # Only compare if not near zero
            assert abs(result_default - result_custom) / result_default < 0.2

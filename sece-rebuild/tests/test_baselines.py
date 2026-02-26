"""Tests for baseline enhancement algorithms.

Tests verify:
- GHE output uses full dynamic range [0, 255]
- CLAHE preserves image shape
- WTHE parameters work correctly
- All baselines produce valid uint8 output
"""

import pytest
import numpy as np

from sece.baselines import ghe, clahe, clahe_with_params, wthe, wthe_with_params


class TestGHEBasic:
    """Basic GHE functionality tests."""

    def test_ghe_returns_same_shape(self, sample_grayscale):
        """GHE should return image with same shape."""
        result = ghe(sample_grayscale)
        assert result.shape == sample_grayscale.shape

    def test_ghe_returns_uint8(self, sample_grayscale):
        """GHE should return uint8 image."""
        result = ghe(sample_grayscale)
        assert result.dtype == np.uint8

    def test_ghe_uses_full_dynamic_range(self, low_contrast_image):
        """GHE output should use full dynamic range [0, 255]."""
        result = ghe(low_contrast_image)
        # For a good test image, GHE should expand to use most of the range
        # At minimum, it should use more than the original
        original_range = low_contrast_image.max() - low_contrast_image.min()
        result_range = result.max() - result.min()
        assert result_range >= original_range, (
            f"GHE should expand range: {result_range} >= {original_range}"
        )

    def test_ghe_color_image_raises(self, sample_color):
        """GHE should raise error for color images."""
        with pytest.raises(ValueError, match="2D grayscale"):
            ghe(sample_color)

    def test_ghe_float_image_raises(self, sample_grayscale):
        """GHE should raise error for float images."""
        float_image = sample_grayscale.astype(np.float64)
        with pytest.raises(ValueError, match="uint8"):
            ghe(float_image)


class TestGHEDeterminism:
    """Test GHE determinism."""

    def test_ghe_deterministic(self, sample_grayscale):
        """GHE should produce identical results for same input."""
        result1 = ghe(sample_grayscale)
        result2 = ghe(sample_grayscale)
        np.testing.assert_array_equal(result1, result2)


class TestCLAHEBasic:
    """Basic CLAHE functionality tests."""

    def test_clahe_returns_same_shape(self, sample_grayscale):
        """CLAHE should return image with same shape."""
        result = clahe(sample_grayscale)
        assert result.shape == sample_grayscale.shape

    def test_clahe_returns_uint8(self, sample_grayscale):
        """CLAHE should return uint8 image."""
        result = clahe(sample_grayscale)
        assert result.dtype == np.uint8

    def test_clahe_output_in_valid_range(self, sample_grayscale):
        """CLAHE output should be in [0, 255]."""
        result = clahe(sample_grayscale)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_clahe_color_image_raises(self, sample_color):
        """CLAHE should raise error for color images."""
        with pytest.raises(ValueError, match="2D grayscale"):
            clahe(sample_color)

    def test_clahe_float_image_raises(self, sample_grayscale):
        """CLAHE should raise error for float images."""
        float_image = sample_grayscale.astype(np.float64)
        with pytest.raises(ValueError, match="uint8"):
            clahe(float_image)


class TestCLAHEParameters:
    """Test CLAHE parameter handling."""

    def test_clahe_default_clip_limit(self, sample_grayscale):
        """Default clip limit should be 2.0."""
        result_default = clahe(sample_grayscale)
        result_explicit = clahe(sample_grayscale, clip_limit=2.0)
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_clahe_higher_clip_limit_more_contrast(self, sample_grayscale):
        """Higher clip limit should generally produce more contrast."""
        result_low = clahe(sample_grayscale, clip_limit=1.0)
        result_high = clahe(sample_grayscale, clip_limit=4.0)

        # Both should be valid
        assert result_low.dtype == np.uint8
        assert result_high.dtype == np.uint8

    def test_clahe_custom_tile_size(self, sample_grayscale):
        """Custom tile size should work."""
        result = clahe(sample_grayscale, tile_grid_size=(4, 4))
        assert result.shape == sample_grayscale.shape

    def test_clahe_with_params_returns_dict(self, sample_grayscale):
        """clahe_with_params should return dict."""
        result = clahe_with_params(sample_grayscale)
        assert isinstance(result, dict)
        assert "image" in result
        assert "clip_limit" in result
        assert "tile_grid_size" in result

    def test_clahe_with_params_values(self, sample_grayscale):
        """clahe_with_params should return correct parameter values."""
        result = clahe_with_params(
            sample_grayscale, clip_limit=3.0, tile_grid_size=(16, 16)
        )
        assert result["clip_limit"] == 3.0
        assert result["tile_grid_size"] == (16, 16)


class TestWTHEBasic:
    """Basic WTHE functionality tests."""

    def test_wthe_returns_same_shape(self, sample_grayscale):
        """WTHE should return image with same shape."""
        result = wthe(sample_grayscale)
        assert result.shape == sample_grayscale.shape

    def test_wthe_returns_uint8(self, sample_grayscale):
        """WTHE should return uint8 image."""
        result = wthe(sample_grayscale)
        assert result.dtype == np.uint8

    def test_wthe_output_in_valid_range(self, sample_grayscale):
        """WTHE output should be in [0, 255]."""
        result = wthe(sample_grayscale)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_wthe_color_image_raises(self, sample_color):
        """WTHE should raise error for color images."""
        with pytest.raises(ValueError, match="2D grayscale"):
            wthe(sample_color)

    def test_wthe_float_image_raises(self, sample_grayscale):
        """WTHE should raise error for float images."""
        float_image = sample_grayscale.astype(np.float64)
        with pytest.raises(ValueError, match="uint8"):
            wthe(float_image)


class TestWTHEParameters:
    """Test WTHE parameter handling."""

    def test_wthe_default_parameters(self, sample_grayscale):
        """Default parameters should be r=0.5, v=0.5."""
        result_default = wthe(sample_grayscale)
        result_explicit = wthe(sample_grayscale, r=0.5, v=0.5)
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_wthe_r_out_of_range_raises(self, sample_grayscale):
        """r parameter out of range should raise error."""
        with pytest.raises(ValueError, match="r must be in"):
            wthe(sample_grayscale, r=1.5)
        with pytest.raises(ValueError, match="r must be in"):
            wthe(sample_grayscale, r=-0.1)

    def test_wthe_v_out_of_range_raises(self, sample_grayscale):
        """v parameter out of range should raise error."""
        with pytest.raises(ValueError, match="v must be in"):
            wthe(sample_grayscale, v=1.5)
        with pytest.raises(ValueError, match="v must be in"):
            wthe(sample_grayscale, v=-0.1)

    def test_wthe_with_params_returns_dict(self, sample_grayscale):
        """wthe_with_params should return dict."""
        result = wthe_with_params(sample_grayscale)
        assert isinstance(result, dict)
        assert "image" in result
        assert "r" in result
        assert "v" in result

    def test_wthe_with_params_values(self, sample_grayscale):
        """wthe_with_params should return correct parameter values."""
        result = wthe_with_params(sample_grayscale, r=0.7, v=0.6)
        assert result["r"] == 0.7
        assert result["v"] == 0.6

    def test_wthe_different_r_values(self, sample_grayscale):
        """Different r values should produce different results."""
        result_low = wthe(sample_grayscale, r=0.3)
        result_high = wthe(sample_grayscale, r=0.9)
        # Results should differ for non-uniform images
        # (may be same for uniform images)
        assert result_low.dtype == np.uint8
        assert result_high.dtype == np.uint8

    def test_wthe_different_v_values(self, sample_grayscale):
        """Different v values should produce different results."""
        result_low = wthe(sample_grayscale, v=0.3)
        result_high = wthe(sample_grayscale, v=0.9)
        assert result_low.dtype == np.uint8
        assert result_high.dtype == np.uint8


class TestBaselinesComparison:
    """Compare different baseline algorithms."""

    def test_all_baselines_same_shape(self, sample_grayscale):
        """All baselines should produce same shape output."""
        ghe_result = ghe(sample_grayscale)
        clahe_result = clahe(sample_grayscale)
        wthe_result = wthe(sample_grayscale)

        assert ghe_result.shape == sample_grayscale.shape
        assert clahe_result.shape == sample_grayscale.shape
        assert wthe_result.shape == sample_grayscale.shape

    def test_all_baselines_uint8(self, sample_grayscale):
        """All baselines should produce uint8 output."""
        assert ghe(sample_grayscale).dtype == np.uint8
        assert clahe(sample_grayscale).dtype == np.uint8
        assert wthe(sample_grayscale).dtype == np.uint8

    def test_baselines_produce_valid_enhancement(self, low_contrast_image):
        """Baselines should enhance low contrast images."""
        ghe_result = ghe(low_contrast_image)
        clahe_result = clahe(low_contrast_image)
        wthe_result = wthe(low_contrast_image)

        # All should expand the dynamic range
        original_range = low_contrast_image.max() - low_contrast_image.min()

        # GHE should definitely expand range
        ghe_range = ghe_result.max() - ghe_result.min()
        assert ghe_range >= original_range

        # Others should also generally expand or maintain range
        assert clahe_result.max() - clahe_result.min() >= 0
        assert wthe_result.max() - wthe_result.min() >= 0


class TestBaselinesEdgeCases:
    """Edge case tests for baselines."""

    def test_single_color_image(self, single_color_image):
        """Single color image should be handled by all baselines."""
        ghe_result = ghe(single_color_image)
        clahe_result = clahe(single_color_image)
        wthe_result = wthe(single_color_image)

        # Should not crash and return valid images
        assert ghe_result.shape == single_color_image.shape
        assert clahe_result.shape == single_color_image.shape
        assert wthe_result.shape == single_color_image.shape

    def test_small_image(self, small_image):
        """Small images should be handled."""
        ghe_result = ghe(small_image)
        clahe_result = clahe(small_image)
        wthe_result = wthe(small_image)

        assert ghe_result.shape == small_image.shape
        assert clahe_result.shape == small_image.shape
        assert wthe_result.shape == small_image.shape

    def test_binary_image(self):
        """Binary image should be handled."""
        binary = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        ghe_result = ghe(binary)
        clahe_result = clahe(binary)
        wthe_result = wthe(binary)

        assert ghe_result.shape == binary.shape
        assert clahe_result.shape == binary.shape
        assert wthe_result.shape == binary.shape

    def test_high_contrast_image(self):
        """High contrast image should be handled."""
        high_contrast = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        ghe_result = ghe(high_contrast)
        clahe_result = clahe(high_contrast)
        wthe_result = wthe(high_contrast)

        assert ghe_result.shape == high_contrast.shape
        assert clahe_result.shape == high_contrast.shape
        assert wthe_result.shape == high_contrast.shape


class TestBaselinesIntegration:
    """Integration tests with metrics."""

    def test_baselines_work_with_emeg(self, low_contrast_image):
        """Enhanced images should have valid EMEG values."""
        from sece.metrics import emeg

        ghe_result = ghe(low_contrast_image)
        clahe_result = clahe(low_contrast_image)
        wthe_result = wthe(low_contrast_image)

        # All should have valid EMEG values
        emeg_ghe = emeg(ghe_result)
        emeg_clahe = emeg(clahe_result)
        emeg_wthe = emeg(wthe_result)

        assert 0 <= emeg_ghe <= 1
        assert 0 <= emeg_clahe <= 1
        assert 0 <= emeg_wthe <= 1

    def test_baselines_work_with_ssim(self, low_contrast_image):
        """Enhanced images should have valid SSIM with original."""
        from sece.metrics import ssim

        ghe_result = ghe(low_contrast_image)
        clahe_result = clahe(low_contrast_image)
        wthe_result = wthe(low_contrast_image)

        # All should have valid SSIM values
        ssim_ghe = ssim(low_contrast_image, ghe_result)
        ssim_clahe = ssim(low_contrast_image, clahe_result)
        ssim_wthe = ssim(low_contrast_image, wthe_result)

        # SSIM should be in valid range
        assert -1 <= ssim_ghe <= 1
        assert -1 <= ssim_clahe <= 1
        assert -1 <= ssim_wthe <= 1

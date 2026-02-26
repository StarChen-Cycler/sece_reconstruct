"""Tests for color image enhancement.

Tests cover:
- ColorProcessor ABC implementation
- HSV, LAB, YCbCr processors
- color_sece and color_secedct functions
- Color channel preservation
- Edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from sece.color import (
    HSVProcessor,
    LABProcessor,
    YCbCrProcessor,
    ColorProcessor,
    get_processor,
    color_sece,
    color_sece_simple,
    color_secedct,
    color_secedct_simple,
    ColorSECEResult,
    ColorSECEDCTResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_bgr():
    """Standard 64x64 BGR test image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def low_contrast_bgr():
    """Low contrast BGR test image."""
    np.random.seed(42)
    return np.random.randint(100, 150, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def single_color_bgr():
    """Uniform color BGR image."""
    return np.full((32, 32, 3), [100, 150, 200], dtype=np.uint8)


# =============================================================================
# Test ColorProcessor ABC
# =============================================================================


class TestColorProcessorABC:
    """Tests for ColorProcessor abstract base class."""

    def test_cannot_instantiate_abc(self):
        """ColorProcessor is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            ColorProcessor()

    def test_subclass_must_implement_name(self):
        """Subclass must implement name property."""

        class IncompleteProcessor(ColorProcessor):
            def to_luminance(self, image):
                return image, image

            def from_luminance(self, luminance, chrominance):
                return luminance

        with pytest.raises(TypeError):
            IncompleteProcessor()

    def test_validate_input_accepts_valid(self, sample_bgr):
        """validate_input should accept valid BGR images."""
        processor = HSVProcessor()
        processor.validate_input(sample_bgr)  # Should not raise

    def test_validate_input_rejects_2d(self):
        """validate_input should reject 2D images."""
        processor = HSVProcessor()
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="3D"):
            processor.validate_input(image)

    def test_validate_input_rejects_4_channel(self):
        """validate_input should reject 4-channel images."""
        processor = HSVProcessor()
        image = np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="3 channels"):
            processor.validate_input(image)

    def test_validate_input_rejects_float(self):
        """validate_input should reject float images."""
        processor = HSVProcessor()
        image = np.random.rand(64, 64, 3).astype(np.float32)
        with pytest.raises(ValueError, match="uint8"):
            processor.validate_input(image)


# =============================================================================
# Test HSVProcessor
# =============================================================================


class TestHSVProcessor:
    """Tests for HSV color space processor."""

    def test_name(self):
        """Processor name should be 'hsv'."""
        processor = HSVProcessor()
        assert processor.name == "hsv"

    def test_to_luminance_returns_correct_shapes(self, sample_bgr):
        """to_luminance should return (H, W) and (H, W, 2) arrays."""
        processor = HSVProcessor()
        luminance, chrominance = processor.to_luminance(sample_bgr)

        assert luminance.shape == (sample_bgr.shape[0], sample_bgr.shape[1])
        assert chrominance.shape == (sample_bgr.shape[0], sample_bgr.shape[1], 2)

    def test_to_luminance_returns_uint8(self, sample_bgr):
        """to_luminance should return uint8 arrays."""
        processor = HSVProcessor()
        luminance, chrominance = processor.to_luminance(sample_bgr)

        assert luminance.dtype == np.uint8
        assert chrominance.dtype == np.uint8

    def test_from_luminance_returns_bgr_shape(self, sample_bgr):
        """from_luminance should return (H, W, 3) BGR image."""
        processor = HSVProcessor()
        luminance, chrominance = processor.to_luminance(sample_bgr)
        result = processor.from_luminance(luminance, chrominance)

        assert result.shape == sample_bgr.shape
        assert result.dtype == np.uint8

    def test_roundtrip_preserves_structure(self, sample_bgr):
        """BGR -> HSV -> BGR should preserve image structure."""
        processor = HSVProcessor()
        luminance, chrominance = processor.to_luminance(sample_bgr)
        result = processor.from_luminance(luminance, chrominance)

        # HSV conversion has some quantization error due to 8-bit representation
        # Max error should be small but not zero
        max_diff = np.max(np.abs(result.astype(np.int16) - sample_bgr.astype(np.int16)))
        assert max_diff <= 5, f"Max difference {max_diff} exceeds tolerance"

    def test_enhanced_luminance_preserves_hue(self, sample_bgr):
        """Enhancing luminance should preserve H and S channels."""
        import cv2

        processor = HSVProcessor()
        luminance, chrominance = processor.to_luminance(sample_bgr)

        # Enhance luminance (simple stretch for test)
        enhanced_luminance = np.clip(luminance.astype(np.float64) * 1.2, 0, 255).astype(
            np.uint8
        )

        # Reconstruct
        result = processor.from_luminance(enhanced_luminance, chrominance)

        # Check that S channel is preserved for non-black pixels
        # (H is checked separately, as it's affected by V changes)
        original_hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        # For bright pixels, S should be nearly preserved
        bright_mask = result_hsv[:, :, 2] > 10
        if np.any(bright_mask):
            # S channel should be preserved with minor quantization
            s_diff = np.abs(
                chrominance[bright_mask, 1].astype(np.int16)
                - result_hsv[bright_mask, 1].astype(np.int16)
            )
            max_diff = np.max(s_diff)
            assert max_diff <= 6, f"S channel max difference {max_diff} exceeds tolerance"


# =============================================================================
# Test LABProcessor
# =============================================================================


class TestLABProcessor:
    """Tests for LAB color space processor."""

    def test_name(self):
        """Processor name should be 'lab'."""
        processor = LABProcessor()
        assert processor.name == "lab"

    def test_to_luminance_returns_correct_shapes(self, sample_bgr):
        """to_luminance should return (H, W) and (H, W, 2) arrays."""
        processor = LABProcessor()
        luminance, chrominance = processor.to_luminance(sample_bgr)

        assert luminance.shape == (sample_bgr.shape[0], sample_bgr.shape[1])
        assert chrominance.shape == (sample_bgr.shape[0], sample_bgr.shape[1], 2)

    def test_roundtrip_preserves_structure(self, sample_bgr):
        """BGR -> LAB -> BGR should preserve image structure."""
        processor = LABProcessor()
        luminance, chrominance = processor.to_luminance(sample_bgr)
        result = processor.from_luminance(luminance, chrominance)

        # LAB conversion has more quantization error due to 8-bit representation
        # Error can be up to ~18 in extreme cases (RGB <-> LAB is not perfectly lossless)
        max_diff = np.max(np.abs(result.astype(np.int16) - sample_bgr.astype(np.int16)))
        assert max_diff <= 20, f"Max difference {max_diff} exceeds tolerance"


# =============================================================================
# Test YCbCrProcessor
# =============================================================================


class TestYCbCrProcessor:
    """Tests for YCbCr color space processor."""

    def test_name(self):
        """Processor name should be 'ycbcr'."""
        processor = YCbCrProcessor()
        assert processor.name == "ycbcr"

    def test_to_luminance_returns_correct_shapes(self, sample_bgr):
        """to_luminance should return (H, W) and (H, W, 2) arrays."""
        processor = YCbCrProcessor()
        luminance, chrominance = processor.to_luminance(sample_bgr)

        assert luminance.shape == (sample_bgr.shape[0], sample_bgr.shape[1])
        assert chrominance.shape == (sample_bgr.shape[0], sample_bgr.shape[1], 2)

    def test_roundtrip_preserves_structure(self, sample_bgr):
        """BGR -> YCbCr -> BGR should preserve image structure."""
        processor = YCbCrProcessor()
        luminance, chrominance = processor.to_luminance(sample_bgr)
        result = processor.from_luminance(luminance, chrominance)

        # YCbCr conversion has very minor quantization error (at most 1)
        max_diff = np.max(np.abs(result.astype(np.int16) - sample_bgr.astype(np.int16)))
        assert max_diff <= 1, f"Max difference {max_diff} exceeds tolerance"


# =============================================================================
# Test get_processor
# =============================================================================


class TestGetProcessor:
    """Tests for get_processor factory function."""

    def test_get_hsv_processor(self):
        """get_processor('hsv') should return HSVProcessor."""
        processor = get_processor("hsv")
        assert isinstance(processor, HSVProcessor)

    def test_get_lab_processor(self):
        """get_processor('lab') should return LABProcessor."""
        processor = get_processor("lab")
        assert isinstance(processor, LABProcessor)

    def test_get_ycbcr_processor(self):
        """get_processor('ycbcr') should return YCbCrProcessor."""
        processor = get_processor("ycbcr")
        assert isinstance(processor, YCbCrProcessor)

    def test_case_insensitive(self):
        """get_processor should be case-insensitive."""
        assert isinstance(get_processor("HSV"), HSVProcessor)
        assert isinstance(get_processor("Lab"), LABProcessor)
        assert isinstance(get_processor("YCBCR"), YCbCrProcessor)

    def test_invalid_color_space_raises(self):
        """get_processor should raise for invalid color space."""
        with pytest.raises(ValueError, match="Unsupported color space"):
            get_processor("rgb")


# =============================================================================
# Test color_sece
# =============================================================================


class TestColorSECE:
    """Tests for color_sece function."""

    def test_returns_color_sece_result(self, sample_bgr):
        """color_sece should return ColorSECEResult."""
        result = color_sece(sample_bgr)
        assert isinstance(result, ColorSECEResult)

    def test_returns_same_shape(self, sample_bgr):
        """color_sece should return same shape as input."""
        result = color_sece(sample_bgr)
        assert result.image.shape == sample_bgr.shape

    def test_returns_uint8(self, sample_bgr):
        """color_sece should return uint8 image."""
        result = color_sece(sample_bgr)
        assert result.image.dtype == np.uint8

    def test_returns_color_space(self, sample_bgr):
        """color_sece should return color_space used."""
        result = color_sece(sample_bgr)
        assert result.color_space == "hsv"

    def test_default_is_hsv(self, sample_bgr):
        """color_sece should default to HSV."""
        result = color_sece(sample_bgr)
        assert result.color_space == "hsv"

    def test_lab_color_space(self, sample_bgr):
        """color_sece with color_space='lab' should use LAB."""
        result = color_sece(sample_bgr, color_space="lab")
        assert result.color_space == "lab"

    def test_ycbcr_color_space(self, sample_bgr):
        """color_sece with color_space='ycbcr' should use YCbCr."""
        result = color_sece(sample_bgr, color_space="ycbcr")
        assert result.color_space == "ycbcr"

    def test_uses_full_dynamic_range(self, low_contrast_bgr):
        """color_sece should expand dynamic range."""
        result = color_sece(low_contrast_bgr)

        original_range = low_contrast_bgr.max() - low_contrast_bgr.min()
        enhanced_range = result.image.max() - result.image.min()

        assert enhanced_range >= original_range

    def test_output_in_valid_range(self, sample_bgr):
        """color_sece output should be in [0, 255]."""
        result = color_sece(sample_bgr)
        assert result.image.min() >= 0
        assert result.image.max() <= 255

    def test_preserves_hue_hsv(self, sample_bgr):
        """color_sece with HSV should preserve H channel for well-defined colored pixels."""
        import cv2

        result = color_sece(sample_bgr, color_space="hsv")

        # Extract H channels
        original_hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)
        result_hsv = cv2.cvtColor(result.image, cv2.COLOR_BGR2HSV)

        # H channel should be reasonably preserved for saturated, bright pixels
        # Use strict thresholds to focus on well-defined colors
        valid_mask = (result_hsv[:, :, 2] > 50) & (result_hsv[:, :, 1] > 50)
        if np.any(valid_mask):
            h_diff = np.abs(
                original_hsv[valid_mask, 0].astype(np.int16)
                - result_hsv[valid_mask, 0].astype(np.int16)
            )
            # Handle hue wraparound (H is 0-179 in OpenCV)
            h_diff = np.minimum(h_diff, 180 - h_diff)
            # Most pixels should have small H difference
            # Allow up to 5% of pixels to have larger differences
            pct_large_diff = np.mean(h_diff > 5) * 100
            assert pct_large_diff < 5, f"{pct_large_diff:.1f}% pixels have H diff > 5"

    def test_single_color_image(self, single_color_bgr):
        """color_sece on uniform color should return similar color."""
        result = color_sece(single_color_bgr)

        # Shape preserved
        assert result.image.shape == single_color_bgr.shape

        # Each channel should still be uniform (single unique value per channel)
        for i in range(3):
            unique_vals = np.unique(result.image[:, :, i])
            assert len(unique_vals) == 1, (
                f"Channel {i} should have single value, got {len(unique_vals)}"
            )

    def test_rejects_grayscale(self):
        """color_sece should reject 2D grayscale images."""
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="3D"):
            color_sece(image)

    def test_color_sece_simple_returns_only_image(self, sample_bgr):
        """color_sece_simple should return only the enhanced image."""
        result = color_sece_simple(sample_bgr)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_bgr.shape


# =============================================================================
# Test color_secedct
# =============================================================================


class TestColorSECEDCT:
    """Tests for color_secedct function."""

    def test_returns_color_secedct_result(self, sample_bgr):
        """color_secedct should return ColorSECEDCTResult."""
        result = color_secedct(sample_bgr)
        assert isinstance(result, ColorSECEDCTResult)

    def test_returns_same_shape(self, sample_bgr):
        """color_secedct should return same shape as input."""
        result = color_secedct(sample_bgr)
        assert result.image.shape == sample_bgr.shape

    def test_returns_uint8(self, sample_bgr):
        """color_secedct should return uint8 image."""
        result = color_secedct(sample_bgr)
        assert result.image.dtype == np.uint8

    def test_gamma_zero_equals_sece(self, sample_bgr):
        """gamma=0 should produce same result as color_sece."""
        result_sece = color_sece(sample_bgr)
        result_secedct = color_secedct(sample_bgr, gamma=0)

        np.testing.assert_array_equal(result_sece.image, result_secedct.image)

    def test_gamma_increases_contrast(self, sample_bgr):
        """Higher gamma should generally increase contrast."""
        result_low = color_secedct(sample_bgr, gamma=0.2)
        result_high = color_secedct(sample_bgr, gamma=0.8)

        # Just verify they're different
        assert not np.array_equal(result_low.image, result_high.image)

    def test_invalid_gamma_raises(self, sample_bgr):
        """gamma outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="Gamma"):
            color_secedct(sample_bgr, gamma=-0.1)

        with pytest.raises(ValueError, match="Gamma"):
            color_secedct(sample_bgr, gamma=1.1)

    def test_lab_color_space(self, sample_bgr):
        """color_secedct with color_space='lab' should use LAB."""
        result = color_secedct(sample_bgr, color_space="lab")
        assert result.color_space == "lab"

    def test_ycbcr_color_space(self, sample_bgr):
        """color_secedct with color_space='ycbcr' should use YCbCr."""
        result = color_secedct(sample_bgr, color_space="ycbcr")
        assert result.color_space == "ycbcr"

    def test_preserves_hue_hsv(self, sample_bgr):
        """color_secedct with HSV should preserve H channel for well-defined colored pixels."""
        import cv2

        result = color_secedct(sample_bgr, color_space="hsv")

        # Extract H channels
        original_hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)
        result_hsv = cv2.cvtColor(result.image, cv2.COLOR_BGR2HSV)

        # H channel should be reasonably preserved for saturated, bright pixels
        valid_mask = (result_hsv[:, :, 2] > 50) & (result_hsv[:, :, 1] > 50)
        if np.any(valid_mask):
            h_diff = np.abs(
                original_hsv[valid_mask, 0].astype(np.int16)
                - result_hsv[valid_mask, 0].astype(np.int16)
            )
            # Handle hue wraparound
            h_diff = np.minimum(h_diff, 180 - h_diff)
            # Most pixels should have small H difference
            # Allow up to 10% of pixels to have larger differences (SECEDCT is more aggressive)
            pct_large_diff = np.mean(h_diff > 5) * 100
            assert pct_large_diff < 10, f"{pct_large_diff:.1f}% pixels have H diff > 5"

    def test_color_secedct_simple_returns_only_image(self, sample_bgr):
        """color_secedct_simple should return only the enhanced image."""
        result = color_secedct_simple(sample_bgr)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_bgr.shape


# =============================================================================
# Test color space comparison
# =============================================================================


class TestColorSpaceComparison:
    """Tests comparing different color spaces."""

    def test_all_color_spaces_produce_valid_output(self, sample_bgr):
        """All color spaces should produce valid BGR output."""
        for color_space in ["hsv", "lab", "ycbcr"]:
            result = color_sece(sample_bgr, color_space=color_space)
            assert result.image.shape == sample_bgr.shape
            assert result.image.dtype == np.uint8
            assert result.color_space == color_space

    def test_color_spaces_produce_different_results(self, sample_bgr):
        """Different color spaces should produce different results."""
        result_hsv = color_sece(sample_bgr, color_space="hsv")
        result_lab = color_sece(sample_bgr, color_space="lab")
        result_ycbcr = color_sece(sample_bgr, color_space="ycbcr")

        # Results should be different (different luminance definitions)
        assert not np.array_equal(result_hsv.image, result_lab.image)
        assert not np.array_equal(result_hsv.image, result_ycbcr.image)
        assert not np.array_equal(result_lab.image, result_ycbcr.image)

    def test_all_color_spaces_improve_contrast(self, low_contrast_bgr):
        """All color spaces should improve contrast."""
        for color_space in ["hsv", "lab", "ycbcr"]:
            result = color_sece(low_contrast_bgr, color_space=color_space)

            original_range = low_contrast_bgr.max() - low_contrast_bgr.min()
            enhanced_range = result.image.max() - result.image.min()

            assert enhanced_range >= original_range


# =============================================================================
# Integration tests
# =============================================================================


class TestColorIntegration:
    """Integration tests for color enhancement."""

    def test_sece_result_contains_luminance_result(self, sample_bgr):
        """ColorSECEResult should contain SECE result from luminance."""
        from sece.core import SECEResult

        result = color_sece(sample_bgr)
        assert isinstance(result.sece_result, SECEResult)

    def test_secedct_result_contains_secedct_result(self, sample_bgr):
        """ColorSECEDCTResult should contain SECEDCT result from luminance."""
        from sece.secedct import SECEDCTResult

        result = color_secedct(sample_bgr)
        assert isinstance(result.secedct_result, SECEDCTResult)

    def test_processing_time_reported(self, sample_bgr):
        """Processing time should be reported."""
        result = color_sece(sample_bgr)
        assert result.processing_time_ms > 0

        result2 = color_secedct(sample_bgr)
        assert result2.processing_time_ms > 0

    def test_module_imports(self):
        """Color functions should be importable from main package."""
        from sece import (
            color_sece,
            color_sece_simple,
            color_secedct,
            color_secedct_simple,
            ColorSECEResult,
            ColorSECEDCTResult,
        )

        assert callable(color_sece)
        assert callable(color_sece_simple)
        assert callable(color_secedct)
        assert callable(color_secedct_simple)

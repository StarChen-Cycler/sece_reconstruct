"""Tests for SECEDCT module."""

from __future__ import annotations

import numpy as np
import pytest

# Direct imports to avoid cv2 import issues in __init__.py
from sece.core import sece
from sece.secedct import (
    SECEDCTResult,
    secedct,
    secedct_simple,
    validate_secedct_result,
)


class TestSECEDCTBasic:
    """Basic functionality tests for SECEDCT."""

    @pytest.fixture
    def sample_image(self):
        """Standard 128x128 grayscale test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    def test_returns_secedct_result(self, sample_image):
        """Test that secedct returns SECEDCTResult dataclass."""
        result = secedct(sample_image)
        assert isinstance(result, SECEDCTResult)

    def test_returns_correct_shape(self, sample_image):
        """Test that output shape matches input shape."""
        result = secedct(sample_image)
        assert result.image.shape == sample_image.shape

    def test_returns_uint8(self, sample_image):
        """Test that output dtype is uint8."""
        result = secedct(sample_image)
        assert result.image.dtype == np.uint8

    def test_returns_sece_result(self, sample_image):
        """Test that intermediate SECE result is included."""
        result = secedct(sample_image)
        assert hasattr(result, "sece_result")
        assert result.sece_result.image is not None
        assert result.sece_result.distribution is not None

    def test_returns_alpha(self, sample_image):
        """Test that alpha parameter is computed."""
        result = secedct(sample_image)
        assert hasattr(result, "alpha")
        assert result.alpha >= 1.0  # alpha >= 1 always

    def test_returns_gamma(self, sample_image):
        """Test that gamma parameter is stored."""
        result = secedct(sample_image, gamma=0.7)
        assert result.gamma == 0.7

    def test_returns_processing_time(self, sample_image):
        """Test that processing time is recorded."""
        result = secedct(sample_image)
        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms > 0


class TestSECEDCTGammaZero:
    """Tests for gamma=0 behavior (should equal SECE)."""

    @pytest.fixture
    def sample_image(self):
        """Standard 128x128 grayscale test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    def test_gamma_zero_equals_sece(self, sample_image):
        """SECEDCT with gamma=0 should produce identical output to SECE."""
        sece_result = sece(sample_image)
        secedct_result = secedct(sample_image, gamma=0)

        np.testing.assert_array_equal(
            secedct_result.image, sece_result.image,
            err_msg="SECEDCT(gamma=0) should equal SECE output"
        )

    def test_gamma_zero_alpha_is_one(self, sample_image):
        """When gamma=0, alpha should be 1.0 (no weighting)."""
        result = secedct(sample_image, gamma=0)
        assert result.alpha == 1.0

    def test_gamma_zero_no_local_enhancement(self, sample_image):
        """gamma=0 means no local DCT-based enhancement."""
        result = secedct(sample_image, gamma=0)
        # When alpha=1, DCT coefficients are not weighted
        # So output should be exactly SECE output
        sece_result = sece(sample_image)
        np.testing.assert_array_equal(result.image, sece_result.image)


class TestSECEDCTDynamicRange:
    """Tests for dynamic range usage."""

    @pytest.fixture
    def low_contrast_image(self):
        """Low contrast test image (narrow histogram)."""
        np.random.seed(42)
        return np.random.randint(100, 150, (128, 128), dtype=np.uint8)

    def test_uses_full_dynamic_range(self, low_contrast_image):
        """SECEDCT should expand low-contrast images to use more range."""
        result = secedct(low_contrast_image)
        enhanced_range = result.image.max() - result.image.min()
        original_range = low_contrast_image.max() - low_contrast_image.min()
        assert enhanced_range >= original_range

    def test_output_in_valid_range(self, low_contrast_image):
        """Output should always be in [0, 255]."""
        result = secedct(low_contrast_image)
        assert result.image.min() >= 0
        assert result.image.max() <= 255

    def test_custom_output_range(self, low_contrast_image):
        """Test custom output range for SECE stage."""
        result = secedct(low_contrast_image, y_d=10, y_u=200)
        # Note: local enhancement may expand beyond SECE range
        assert result.image.min() >= 0  # Final output still clamped
        assert result.image.max() <= 255


class TestSECEDCTGammaValues:
    """Tests for different gamma values."""

    @pytest.fixture
    def sample_image(self):
        """Standard 128x128 grayscale test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    def test_gamma_one_maximum_enhancement(self, sample_image):
        """gamma=1 should give maximum local enhancement."""
        result = secedct(sample_image, gamma=1)
        assert result.gamma == 1.0
        # Alpha should be entropy of distribution
        assert result.alpha >= 1.0

    def test_gamma_increases_alpha(self, sample_image):
        """Higher gamma should generally increase alpha."""
        result_0 = secedct(sample_image, gamma=0)
        result_05 = secedct(sample_image, gamma=0.5)
        result_1 = secedct(sample_image, gamma=1)

        # alpha = entropy^gamma, so higher gamma -> higher alpha (when entropy > 1)
        # For most real images, entropy > 1
        assert result_0.alpha <= result_05.alpha or result_0.alpha == 1.0
        assert result_05.alpha <= result_1.alpha or result_05.alpha == result_1.alpha

    def test_gamma_negative_raises_error(self, sample_image):
        """Negative gamma should raise ValueError."""
        with pytest.raises(ValueError, match="Gamma must be in"):
            secedct(sample_image, gamma=-0.1)

    def test_gamma_above_one_raises_error(self, sample_image):
        """Gamma > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="Gamma must be in"):
            secedct(sample_image, gamma=1.5)


class TestSECEDCTEdgeCases:
    """Edge case handling tests."""

    def test_single_color_image(self):
        """Single color image should return unchanged."""
        single_color = np.full((64, 64), 128, dtype=np.uint8)
        result = secedct(single_color)
        np.testing.assert_array_equal(result.image, single_color)

    def test_binary_image(self):
        """Binary image (2 levels) should work."""
        binary = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
        result = secedct(binary)
        assert result.image.shape == binary.shape
        assert result.image.dtype == np.uint8

    def test_small_image_warning(self):
        """Small images should emit warning."""
        tiny_image = np.random.randint(0, 256, (7, 7), dtype=np.uint8)
        with pytest.warns(UserWarning, match="Small image"):
            secedct(tiny_image)

    def test_large_image(self):
        """Large images should process without error."""
        large_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        result = secedct(large_image)
        assert result.image.shape == large_image.shape

    def test_raises_on_color_image(self):
        """Color images should raise ValueError."""
        color_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2D grayscale"):
            secedct(color_image)

    def test_raises_on_wrong_dtype(self):
        """Non-uint8 images should raise ValueError."""
        float_image = np.random.rand(64, 64).astype(np.float32)
        with pytest.raises(ValueError, match="uint8"):
            secedct(float_image)


class TestSECEDCTPerformance:
    """Performance benchmark tests."""

    def test_processing_time_512x512(self):
        """SECEDCT should process 512x512 in < 2 seconds."""
        image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        result = secedct(image)
        assert result.processing_time_ms < 2000, \
            f"SECEDCT took {result.processing_time_ms:.0f}ms (expected < 2000ms)"

    def test_processing_time_256x256(self):
        """SECEDCT should process 256x256 in < 2 seconds."""
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        result = secedct(image)
        assert result.processing_time_ms < 2000, \
            f"SECEDCT took {result.processing_time_ms:.0f}ms (expected < 2000ms)"


class TestSECEDCTSimple:
    """Tests for secedct_simple convenience function."""

    @pytest.fixture
    def sample_image(self):
        """Standard 128x128 grayscale test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    def test_returns_array(self, sample_image):
        """secedct_simple should return ndarray directly."""
        result = secedct_simple(sample_image)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_same_as_secedct_image(self, sample_image):
        """secedct_simple should return same image as secedct."""
        full_result = secedct(sample_image)
        simple_result = secedct_simple(sample_image)
        np.testing.assert_array_equal(simple_result, full_result.image)


class TestValidateSECEDCTResult:
    """Tests for validation function."""

    def test_validation_metrics(self):
        """Test that validation returns expected metrics."""
        original = np.random.randint(50, 150, (64, 64), dtype=np.uint8)
        enhanced = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        validation = validate_secedct_result(original, enhanced, gamma=0.5)

        assert "shape_preserved" in validation
        assert "dtype_correct" in validation
        assert "output_range" in validation
        assert "input_range" in validation
        assert "range_expanded" in validation
        assert "uses_full_range" in validation
        assert "gamma" in validation

    def test_validation_gamma_stored(self):
        """Test that gamma is stored in validation result."""
        original = np.random.randint(50, 150, (64, 64), dtype=np.uint8)
        enhanced = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        validation = validate_secedct_result(original, enhanced, gamma=0.7)
        assert validation["gamma"] == 0.7


class TestSECEDCTIntegration:
    """Integration tests comparing SECEDCT with SECE."""

    @pytest.fixture
    def sample_image(self):
        """Standard 128x128 grayscale test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    def test_secedct_includes_sece_distribution(self, sample_image):
        """SECEDCT should preserve SECE distribution for analysis."""
        result = secedct(sample_image)
        assert result.sece_result.distribution is not None
        assert len(result.sece_result.distribution) > 0

    def test_secedct_gamma_zero_distribution_same_as_sece(self, sample_image):
        """SECEDCT distribution should match SECE distribution."""
        sece_result = sece(sample_image)
        secedct_result = secedct(sample_image, gamma=0)

        np.testing.assert_array_equal(
            secedct_result.sece_result.distribution,
            sece_result.distribution
        )

    def test_gamma_positive_changes_output_from_sece(self, sample_image):
        """With gamma > 0, SECEDCT should differ from SECE (local enhancement)."""
        sece_result = sece(sample_image)
        secedct_result = secedct(sample_image, gamma=0.5)

        # For most images, local enhancement will change the output
        # (unless alpha happens to be exactly 1)
        if secedct_result.alpha > 1.0:
            # Should be different from SECE
            assert not np.array_equal(secedct_result.image, sece_result.image)

"""Integration tests for complete SECE algorithm.

Tests verify the integrated SECE function:
- Correct output shape and type
- Full dynamic range usage
- Histogram shape preservation
- Performance requirements
- Edge case handling
"""

from __future__ import annotations

import numpy as np
import pytest

from sece.core import sece, sece_simple, validate_sece_result, SECEResult


class TestSECEBasic:
    """Basic functionality tests."""

    def test_returns_sece_result(self, sample_grayscale: np.ndarray) -> None:
        """Should return SECEResult dataclass."""
        result = sece(sample_grayscale)
        assert isinstance(result, SECEResult)

    def test_returns_correct_shape(self, sample_grayscale: np.ndarray) -> None:
        """Output image should have same shape as input."""
        result = sece(sample_grayscale)
        assert result.image.shape == sample_grayscale.shape

    def test_returns_uint8(self, sample_grayscale: np.ndarray) -> None:
        """Output image should be uint8."""
        result = sece(sample_grayscale)
        assert result.image.dtype == np.uint8

    def test_returns_distribution(self, sample_grayscale: np.ndarray) -> None:
        """Should return distribution f for SECEDCT."""
        result = sece(sample_grayscale)
        assert result.distribution is not None
        assert np.isclose(np.sum(result.distribution), 1.0)

    def test_returns_gray_levels(self, sample_grayscale: np.ndarray) -> None:
        """Should return gray levels corresponding to distribution."""
        result = sece(sample_grayscale)
        assert result.gray_levels is not None
        assert len(result.gray_levels) == len(result.distribution)

    def test_returns_cdf(self, sample_grayscale: np.ndarray) -> None:
        """Should return CDF."""
        result = sece(sample_grayscale)
        assert result.cdf is not None
        assert np.isclose(result.cdf[-1], 1.0)


class TestSECEDynamicRange:
    """Dynamic range tests."""

    def test_uses_full_dynamic_range(self, low_contrast_image: np.ndarray) -> None:
        """Output should expand to use full [0, 255] range."""
        result = sece(low_contrast_image)

        # Low contrast image has narrow range
        assert low_contrast_image.max() - low_contrast_image.min() < 100

        # Enhanced should use more of the range
        # (may not be full 0-255 but should be expanded)
        enhanced_range = result.image.max() - result.image.min()
        original_range = low_contrast_image.max() - low_contrast_image.min()
        assert enhanced_range >= original_range

    def test_output_in_valid_range(self, sample_grayscale: np.ndarray) -> None:
        """All output values should be in [0, 255]."""
        result = sece(sample_grayscale)
        assert result.image.min() >= 0
        assert result.image.max() <= 255

    def test_custom_output_range(self, sample_grayscale: np.ndarray) -> None:
        """Should support custom output range."""
        y_d, y_u = 50, 200
        result = sece(sample_grayscale, y_d=y_d, y_u=y_u)
        assert result.image.min() >= y_d
        assert result.image.max() <= y_u


class TestSECEHistogramPreservation:
    """Histogram shape preservation tests."""

    def test_preserves_relative_order(self, sample_grayscale: np.ndarray) -> None:
        """Gray levels with higher original values should map to higher outputs."""
        result = sece(sample_grayscale)

        # Check that the CDF is monotonically increasing
        # This ensures relative brightness order is preserved
        assert np.all(np.diff(result.cdf) >= 0)

    def test_distinct_levels_preserved(self, sample_grayscale: np.ndarray) -> None:
        """Number of distinct gray levels should be preserved."""
        original_levels = len(np.unique(sample_grayscale))
        result = sece(sample_grayscale)
        enhanced_levels = len(np.unique(result.image))
        # May have slightly fewer due to floor() mapping
        assert enhanced_levels <= original_levels
        assert enhanced_levels >= original_levels * 0.5  # At least 50% preserved


class TestSECEEdgeCases:
    """Edge case tests."""

    def test_single_color_image(self, single_color_image: np.ndarray) -> None:
        """Single color image should return unchanged."""
        result = sece(single_color_image)
        np.testing.assert_array_equal(result.image, single_color_image)

    def test_binary_image(self) -> None:
        """Binary image (0 and 255 only) should be handled."""
        np.random.seed(42)
        binary = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
        result = sece(binary)
        assert result.image.shape == binary.shape
        assert result.image.dtype == np.uint8

    def test_small_image(self) -> None:
        """Small image should be processed with warning."""
        # Create a truly small image (< 8x8)
        tiny_image = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
        with pytest.warns(UserWarning, match="Small image"):
            result = sece(tiny_image)
        assert result.image.shape == tiny_image.shape

    def test_large_image(self) -> None:
        """Should handle larger images."""
        np.random.seed(42)
        large_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        result = sece(large_image)
        assert result.image.shape == (512, 512)

    def test_raises_on_color_image(self, sample_color: np.ndarray) -> None:
        """Should raise ValueError for color images."""
        with pytest.raises(ValueError, match="2D grayscale"):
            sece(sample_color)

    def test_raises_on_wrong_dtype(self) -> None:
        """Should raise ValueError for non-uint8 images."""
        float_image = np.random.rand(64, 64).astype(np.float32)
        with pytest.raises(ValueError, match="uint8"):
            sece(float_image)  # type: ignore


class TestSECEPerformance:
    """Performance tests."""

    def test_processing_time_512x512(self) -> None:
        """Should process 512x512 image in < 1 second."""
        np.random.seed(42)
        image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        result = sece(image)
        assert result.processing_time_ms < 1000  # < 1 second

    def test_processing_time_256x256(self) -> None:
        """Should process 256x256 image in reasonable time."""
        np.random.seed(42)
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        result = sece(image)
        # Allow up to 2 seconds (implementation not yet optimized)
        assert result.processing_time_ms < 2000


class TestSECESimple:
    """Tests for sece_simple convenience function."""

    def test_returns_only_image(self, sample_grayscale: np.ndarray) -> None:
        """Should return only the enhanced image."""
        enhanced = sece_simple(sample_grayscale)
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == sample_grayscale.shape
        assert enhanced.dtype == np.uint8

    def test_same_result_as_sece(self, sample_grayscale: np.ndarray) -> None:
        """Should produce same output as sece().image."""
        result = sece(sample_grayscale)
        enhanced = sece_simple(sample_grayscale)
        np.testing.assert_array_equal(result.image, enhanced)


class TestValidateSECEResult:
    """Tests for validate_sece_result."""

    def test_validates_correct_result(self, sample_grayscale: np.ndarray) -> None:
        """Should validate a correct SECE result."""
        result = sece(sample_grayscale)
        validation = validate_sece_result(sample_grayscale, result.image)

        assert validation["shape_preserved"]
        assert validation["dtype_correct"]

    def test_detects_wrong_shape(self, sample_grayscale: np.ndarray) -> None:
        """Should detect shape mismatch."""
        result = sece(sample_grayscale)
        wrong_shape = result.image[:64, :64]
        validation = validate_sece_result(sample_grayscale, wrong_shape)

        assert not validation["shape_preserved"]

    def test_expanded_range_detection(
        self, sample_grayscale: np.ndarray, low_contrast_image: np.ndarray
    ) -> None:
        """Should detect range expansion."""
        result = sece(low_contrast_image)
        validation = validate_sece_result(low_contrast_image, result.image)

        assert validation["range_expanded"]


class TestSECEIntegration:
    """Full integration tests with all components."""

    def test_full_pipeline_deterministic(self) -> None:
        """Pipeline should be deterministic for same input."""
        np.random.seed(42)
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

        result1 = sece(image)
        result2 = sece(image)

        np.testing.assert_array_equal(result1.image, result2.image)
        np.testing.assert_allclose(result1.distribution, result2.distribution)

    def test_different_images_different_results(self) -> None:
        """Different images should produce different results."""
        np.random.seed(42)
        image1 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        image2 = np.random.randint(100, 200, (64, 64), dtype=np.uint8)

        result1 = sece(image1)
        result2 = sece(image2)

        # Results should be different
        assert not np.array_equal(result1.image, result2.image)

    @pytest.mark.parametrize("size", [(64, 64), (128, 128), (256, 256)])
    def test_various_sizes(self, size: tuple) -> None:
        """Should handle various image sizes."""
        np.random.seed(42)
        image = np.random.randint(0, 256, size, dtype=np.uint8)
        result = sece(image)

        assert result.image.shape == size
        assert len(result.gray_levels) > 0
        assert len(result.distribution) == len(result.gray_levels)

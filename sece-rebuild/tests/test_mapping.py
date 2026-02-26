"""Unit tests for gray level mapping function.

Tests verify formula (7) from the SECE paper:
    y_k = floor(F_k * (y_u - y_d) + y_d)
"""

from __future__ import annotations

import numpy as np
import pytest

from sece.mapping import (
    apply_mapping_to_image,
    compute_mapping,
    validate_mapping,
)


class TestComputeMapping:
    """Tests for compute_mapping."""

    def test_returns_correct_shape(self) -> None:
        """Output should have same shape as input F."""
        F = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = compute_mapping(F)
        assert y.shape == F.shape

    def test_returns_uint8(self) -> None:
        """Output should be uint8 for image compatibility."""
        F = np.array([0.0, 0.5, 1.0])
        y = compute_mapping(F)
        assert y.dtype == np.uint8

    def test_default_range_0_to_255(self) -> None:
        """Default output range should be [0, 255]."""
        F = np.array([0.0, 0.5, 1.0])
        y = compute_mapping(F)
        assert y[0] >= 0
        assert y[-1] <= 255

    def test_formula_7_correctness(self) -> None:
        """Verify formula (7): y_k = floor(F_k * (y_u - y_d) + y_d)."""
        F = np.array([0.1, 0.3, 0.6, 1.0])
        y = compute_mapping(F, y_d=0, y_u=255)

        # Expected: floor(F * 255)
        expected = np.floor(F * 255).astype(np.uint8)
        np.testing.assert_array_equal(y, expected)

    def test_cdf_0_maps_to_y_d(self) -> None:
        """F=0 should map to y_d."""
        F = np.array([0.0, 0.5, 1.0])
        y = compute_mapping(F, y_d=10, y_u=100)
        assert y[0] == 10

    def test_cdf_1_maps_to_y_u(self) -> None:
        """F=1 should map to y_u."""
        F = np.array([0.0, 0.5, 1.0])
        y = compute_mapping(F, y_d=10, y_u=100)
        assert y[-1] == 100

    def test_custom_output_range(self) -> None:
        """Test with custom y_d and y_u."""
        F = np.array([0.0, 0.5, 1.0])
        y = compute_mapping(F, y_d=50, y_u=200)

        # F=0 maps to 50, F=0.5 maps to ~125, F=1 maps to 200
        assert y[0] == 50
        assert y[-1] == 200
        # Check middle value: floor(0.5 * 150 + 50) = floor(125) = 125
        assert y[1] == 125

    def test_output_in_valid_range(self) -> None:
        """All output values should be in [y_d, y_u]."""
        F = np.linspace(0, 1, 100)
        y = compute_mapping(F, y_d=0, y_u=255)
        assert np.all(y >= 0)
        assert np.all(y <= 255)

    def test_monotonic_non_decreasing(self) -> None:
        """Output should be non-decreasing since CDF is monotonic."""
        F = np.array([0.1, 0.2, 0.5, 0.8, 1.0])
        y = compute_mapping(F)
        assert np.all(np.diff(y) >= 0)

    def test_single_gray_level(self) -> None:
        """K=1 edge case: single CDF value."""
        F = np.array([1.0])  # CDF for single level is 1
        y = compute_mapping(F)
        assert y[0] == 255  # Maps to upper bound

    def test_uniform_cdf(self) -> None:
        """Uniform CDF should produce evenly spaced outputs."""
        K = 10
        F = np.linspace(0, 1, K)
        y = compute_mapping(F)

        # Differences should be roughly equal (allowing for floor rounding)
        diffs = np.diff(y.astype(np.int64))
        assert np.std(diffs) <= 2  # Small variance due to rounding

    def test_raises_on_empty_array(self) -> None:
        """Should raise ValueError for empty F array."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_mapping(np.array([]))

    def test_raises_on_invalid_range(self) -> None:
        """Should raise ValueError if y_d >= y_u."""
        with pytest.raises(ValueError, match="must be less than"):
            compute_mapping(np.array([0.5]), y_d=100, y_u=50)

    def test_raises_on_cdf_out_of_range(self) -> None:
        """Should raise ValueError if F values outside [0, 1]."""
        with pytest.raises(ValueError, match="must be in range"):
            compute_mapping(np.array([-0.1, 0.5]))
        with pytest.raises(ValueError, match="must be in range"):
            compute_mapping(np.array([0.5, 1.5]))


class TestApplyMappingToImage:
    """Tests for apply_mapping_to_image."""

    def test_simple_mapping(self) -> None:
        """Test basic gray level mapping."""
        image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        gray_levels = np.array([0, 1, 2, 3])
        output_levels = np.array([10, 60, 150, 255], dtype=np.uint8)

        result = apply_mapping_to_image(image, gray_levels, output_levels)

        expected = np.array([[10, 60], [150, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_preserves_shape(self) -> None:
        """Output image should have same shape as input."""
        image = np.random.randint(0, 4, (64, 64), dtype=np.uint8)
        gray_levels = np.array([0, 1, 2, 3])
        output_levels = np.array([50, 100, 150, 200], dtype=np.uint8)

        result = apply_mapping_to_image(image, gray_levels, output_levels)
        assert result.shape == image.shape

    def test_all_same_gray_level(self) -> None:
        """Image with all same gray level should map consistently."""
        image = np.full((10, 10), 128, dtype=np.uint8)
        gray_levels = np.array([128])
        output_levels = np.array([200], dtype=np.uint8)

        result = apply_mapping_to_image(image, gray_levels, output_levels)
        np.testing.assert_array_equal(result, np.full((10, 10), 200, dtype=np.uint8))

    def test_raises_on_length_mismatch(self) -> None:
        """Should raise if gray_levels and output_levels have different lengths."""
        image = np.array([[0, 1]], dtype=np.uint8)
        gray_levels = np.array([0, 1, 2])  # 3 levels
        output_levels = np.array([10, 20], dtype=np.uint8)  # 2 levels

        with pytest.raises(ValueError, match="must have same length"):
            apply_mapping_to_image(image, gray_levels, output_levels)

    def test_raises_on_missing_gray_level(self) -> None:
        """Should raise if image has gray level not in mapping."""
        image = np.array([[0, 1, 5]], dtype=np.uint8)  # Has 5
        gray_levels = np.array([0, 1])  # Missing 5
        output_levels = np.array([10, 20], dtype=np.uint8)

        with pytest.raises(ValueError, match="not in gray_levels"):
            apply_mapping_to_image(image, gray_levels, output_levels)


class TestValidateMapping:
    """Tests for validate_mapping."""

    def test_valid_mapping(self) -> None:
        """Valid mapping should pass validation."""
        F = np.array([0.1, 0.3, 0.6, 1.0])
        y = compute_mapping(F)
        assert validate_mapping(y)

    def test_rejects_out_of_range(self) -> None:
        """Should reject values outside [y_d, y_u]."""
        # Test with a value above y_u (but within uint8 range)
        y = np.array([10, 50, 200], dtype=np.uint8)
        # This should fail validation against y_u=100
        assert not validate_mapping(y, y_d=0, y_u=100)

    def test_rejects_decreasing(self) -> None:
        """Should reject non-monotonic mapping."""
        y = np.array([100, 50, 200], dtype=np.uint8)  # 50 < 100
        assert not validate_mapping(y)

    def test_custom_range_validation(self) -> None:
        """Validation should respect custom y_d, y_u."""
        F = np.array([0.0, 0.5, 1.0])
        y = compute_mapping(F, y_d=50, y_u=200)
        assert validate_mapping(y, y_d=50, y_u=200)

        # Should fail if checking against default range
        assert not validate_mapping(y, y_d=0, y_u=100)


class TestMappingIntegration:
    """Integration tests with distribution function."""

    def test_full_pipeline_from_entropy_to_mapping(self) -> None:
        """Test complete pipeline from entropy to output mapping."""
        from sece.distribution import compute_distribution_function

        # Simulate spatial entropy values
        S = np.array([1.0, 2.0, 3.0])

        # Compute distribution
        f, F = compute_distribution_function(S)

        # Compute mapping
        y = compute_mapping(F)

        # Verify properties
        assert y.dtype == np.uint8
        assert np.all(y >= 0)
        assert np.all(y <= 255)
        assert np.all(np.diff(y) >= 0)

    @pytest.mark.parametrize("K", [1, 2, 10, 50, 256])
    def test_various_gray_level_counts(self, K: int) -> None:
        """Test mapping with various numbers of gray levels."""
        from sece.distribution import compute_distribution_function

        np.random.seed(42)
        S = np.random.uniform(0.5, 5.0, K)
        f, F = compute_distribution_function(S)
        y = compute_mapping(F)

        assert y.dtype == np.uint8
        assert len(y) == K
        assert np.all(y >= 0)
        assert np.all(y <= 255)

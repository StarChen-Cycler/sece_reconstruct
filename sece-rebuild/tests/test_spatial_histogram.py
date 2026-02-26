"""Tests for spatial histogram computation."""

from __future__ import annotations

import numpy as np
import pytest

from sece.spatial_histogram import (
    compute_all_spatial_histograms,
    compute_all_spatial_entropies,
    compute_grid_size,
    compute_spatial_entropy,
    compute_spatial_histogram,
)


class TestComputeGridSize:
    """Tests for compute_grid_size function."""

    def test_square_image(self) -> None:
        """Square image should produce square grid."""
        M, N = compute_grid_size(256, (512, 512))
        assert M == N
        assert M * N <= 256  # Total grids <= K

    def test_aspect_ratio_preserved(self) -> None:
        """Grid M/N should approximately match image H/W."""
        H, W = 480, 640
        K = 100
        M, N = compute_grid_size(K, (H, W))

        # Check that M/N approximates H/W
        image_ratio = H / W
        grid_ratio = M / N
        # Allow some tolerance for floor operations
        assert abs(grid_ratio - image_ratio) < 0.2

    def test_landscape_image(self) -> None:
        """Landscape image (W > H) should have N > M."""
        M, N = compute_grid_size(100, (480, 640))
        assert N >= M

    def test_portrait_image(self) -> None:
        """Portrait image (H > W) should have M > N."""
        M, N = compute_grid_size(100, (640, 480))
        assert M >= N

    def test_minimum_grid_size(self) -> None:
        """Grid size should be at least 1x1."""
        M, N = compute_grid_size(1, (8, 8))
        assert M >= 1
        assert N >= 1

    def test_large_K(self) -> None:
        """Large K should produce larger grid."""
        M1, N1 = compute_grid_size(100, (256, 256))
        M2, N2 = compute_grid_size(256, (256, 256))
        assert M2 * N2 > M1 * N1

    @pytest.mark.parametrize(
        "K,shape,expected_range",
        [
            (10, (32, 32), (2, 5)),  # Small image, small K
            (100, (128, 128), (8, 12)),  # Medium
            (256, (512, 512), (14, 18)),  # Large, full range
        ],
    )
    def test_grid_size_reasonable(
        self, K: int, shape: tuple[int, int], expected_range: tuple[int, int]
    ) -> None:
        """Grid dimensions should be reasonable for given K."""
        M, N = compute_grid_size(K, shape)
        min_val, max_val = expected_range
        assert min_val <= M <= max_val
        assert min_val <= N <= max_val


class TestComputeSpatialHistogram:
    """Tests for compute_spatial_histogram function."""

    @pytest.fixture
    def simple_3x3_image(self) -> np.ndarray:
        """Simple 3x3 image with known values."""
        return np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.uint8,
        )

    def test_basic_computation(self, simple_3x3_image: np.ndarray) -> None:
        """Basic histogram computation should work."""
        h = compute_spatial_histogram(simple_3x3_image, 0, 3, 3)
        assert h.shape == (3, 3)
        assert np.all(h >= 0)  # All values non-negative
        assert np.all(h <= 1)  # All values <= 1 (normalized)

    def test_gray_level_0_distribution(
        self, simple_3x3_image: np.ndarray
    ) -> None:
        """Gray level 0 should be concentrated in left column."""
        h = compute_spatial_histogram(simple_3x3_image, 0, 3, 3)
        # Image is [[0,0,1], [0,1,1], [1,1,1]]
        # Top-left cell (0,0) has value 0, should have proportion 1.0
        assert h[0, 0] == 1.0
        # Top-middle cell (0,1) has value 0, should have proportion 1.0
        assert h[0, 1] == 1.0
        # Middle-left cell (1,0) has value 0, should have proportion 1.0
        assert h[1, 0] == 1.0
        # Bottom-left cell (2,0) has value 1, should have proportion 0.0
        assert h[2, 0] == 0.0

    def test_gray_level_1_distribution(
        self, simple_3x3_image: np.ndarray
    ) -> None:
        """Gray level 1 should be in specific cells."""
        h = compute_spatial_histogram(simple_3x3_image, 1, 3, 3)
        assert h.shape == (3, 3)
        # Top-right cell has value 1
        assert h[0, 2] == 1.0
        # Middle row has 1s in cols 1 and 2
        assert h[1, 1] == 1.0
        assert h[1, 2] == 1.0

    def test_invalid_grid_size_raises(self, simple_3x3_image: np.ndarray) -> None:
        """Invalid grid size should raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            compute_spatial_histogram(simple_3x3_image, 0, 0, 3)

        with pytest.raises(ValueError, match="must be >= 1"):
            compute_spatial_histogram(simple_3x3_image, 0, 3, -1)

    def test_non_2d_image_raises(self) -> None:
        """3D color image should raise ValueError."""
        color_image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be 2D"):
            compute_spatial_histogram(color_image, 0, 2, 2)

    def test_uniform_image(self) -> None:
        """Uniform image should have equal distribution in all cells."""
        uniform = np.full((6, 6), 128, dtype=np.uint8)
        h = compute_spatial_histogram(uniform, 128, 2, 2)
        # All cells should have proportion 1.0
        assert np.allclose(h, 1.0)

    def test_empty_gray_level(self) -> None:
        """Gray level not in image should give zero histogram."""
        image = np.zeros((6, 6), dtype=np.uint8)
        h = compute_spatial_histogram(image, 255, 2, 2)
        assert np.allclose(h, 0.0)

    def test_normalization(self) -> None:
        """Histogram values should be normalized by cell area."""
        # 4x4 image with all 1s
        image = np.ones((4, 4), dtype=np.uint8)
        h = compute_spatial_histogram(image, 1, 2, 2)
        # Each 2x2 cell should have proportion 1.0
        assert np.allclose(h, 1.0)


class TestComputeAllSpatialHistograms:
    """Tests for compute_all_spatial_histograms function."""

    def test_returns_all_gray_levels(self) -> None:
        """Should return histograms for all distinct gray levels."""
        image = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.uint8)
        hists, levels, M, N = compute_all_spatial_histograms(image, 3, 3)

        assert len(levels) == 3
        assert set(levels) == {0, 1, 2}
        assert hists.shape[0] == 3

    def test_grid_size_computed_if_not_provided(self) -> None:
        """Grid size should be computed from K and shape if not given."""
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        hists, levels, M, N = compute_all_spatial_histograms(image)

        assert M >= 1
        assert N >= 1
        assert hists.shape[1] == M
        assert hists.shape[2] == N

    def test_custom_grid_size(self) -> None:
        """Custom grid size should be used."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        hists, levels, M, N = compute_all_spatial_histograms(image, M=4, N=4)

        assert M == 4
        assert N == 4
        assert hists.shape == (len(levels), 4, 4)

    def test_invalid_custom_grid_raises(self) -> None:
        """Invalid custom grid size should raise ValueError."""
        image = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be >= 1"):
            compute_all_spatial_histograms(image, M=0, N=4)


class TestComputeSpatialEntropy:
    """Tests for compute_spatial_entropy function."""

    def test_uniform_distribution_high_entropy(self) -> None:
        """Uniform distribution should have maximum entropy."""
        # Uniform distribution across 4 cells
        h = np.full((2, 2), 0.25)
        entropy = compute_spatial_entropy(h)
        # Maximum entropy for 4 cells is log2(4) = 2
        assert abs(entropy - 2.0) < 0.01

    def test_concentrated_low_entropy(self) -> None:
        """Concentrated distribution should have low entropy."""
        # All mass in one cell
        h = np.array([[1.0, 0.0], [0.0, 0.0]])
        entropy = compute_spatial_entropy(h)
        assert entropy < 0.1  # Should be very close to 0

    def test_zero_histogram_returns_zero(self) -> None:
        """Empty histogram should return 0 entropy."""
        h = np.zeros((2, 2))
        entropy = compute_spatial_entropy(h)
        assert entropy == 0.0

    def test_entropy_non_negative(self) -> None:
        """Entropy should always be non-negative."""
        h = np.random.rand(3, 3)
        h = h / h.sum()  # Normalize
        entropy = compute_spatial_entropy(h)
        assert entropy >= 0

    def test_partial_distribution(self) -> None:
        """Distribution spread across some cells."""
        # Spread across 2 of 4 cells
        h = np.array([[0.5, 0.5], [0.0, 0.0]])
        entropy = compute_spatial_entropy(h)
        # Entropy should be log2(2) = 1
        assert abs(entropy - 1.0) < 0.01


class TestComputeAllSpatialEntropies:
    """Tests for compute_all_spatial_entropies function."""

    def test_returns_correct_shape(self) -> None:
        """Should return array with shape (K,)."""
        histograms = np.random.rand(10, 4, 4)
        # Normalize each histogram
        for k in range(10):
            histograms[k] = histograms[k] / histograms[k].sum()

        entropies = compute_all_spatial_entropies(histograms)
        assert entropies.shape == (10,)

    def test_all_entropies_non_negative(self) -> None:
        """All entropies should be non-negative."""
        histograms = np.random.rand(5, 3, 3)
        for k in range(5):
            histograms[k] = histograms[k] / histograms[k].sum()

        entropies = compute_all_spatial_entropies(histograms)
        assert np.all(entropies >= 0)


class TestIntegration:
    """Integration tests for spatial histogram workflow."""

    def test_full_workflow(self) -> None:
        """Test complete workflow from image to entropies."""
        # Create test image
        image = np.random.randint(0, 50, (64, 64), dtype=np.uint8)

        # Compute grid size
        K = len(np.unique(image))
        M, N = compute_grid_size(K, image.shape)

        # Compute all histograms
        histograms, levels, M_used, N_used = compute_all_spatial_histograms(
            image, M, N
        )

        # Compute entropies
        entropies = compute_all_spatial_entropies(histograms)

        # Verify results
        assert len(levels) == K
        assert len(entropies) == K
        assert np.all(entropies >= 0)

    def test_reproducibility(self) -> None:
        """Results should be reproducible for same input."""
        image = np.random.RandomState(42).randint(0, 100, (32, 32), dtype=np.uint8)

        # First run
        hists1, levels1, M1, N1 = compute_all_spatial_histograms(image, 4, 4)
        entropies1 = compute_all_spatial_entropies(hists1)

        # Second run
        hists2, levels2, M2, N2 = compute_all_spatial_histograms(image, 4, 4)
        entropies2 = compute_all_spatial_entropies(hists2)

        np.testing.assert_array_equal(levels1, levels2)
        np.testing.assert_array_almost_equal(hists1, hists2)
        np.testing.assert_array_almost_equal(entropies1, entropies2)

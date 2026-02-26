"""Sanity tests for SECE package."""

from __future__ import annotations

import numpy as np
import pytest
from packaging import version


class TestPackageImport:
    """Tests for package import and version."""

    def test_import_sece(self) -> None:
        """SECE package should be importable."""
        import sece

        assert sece is not None

    def test_version_exists(self) -> None:
        """SECE should have __version__ attribute."""
        import sece

        assert hasattr(sece, "__version__")
        assert isinstance(sece.__version__, str)


class TestDependencies:
    """Tests for core dependencies."""

    def test_numpy_available(self) -> None:
        """NumPy should be available."""
        import numpy as np

        assert version.parse(np.__version__) >= version.parse("1.20")

    def test_scipy_available(self) -> None:
        """SciPy should be available."""
        import scipy

        assert version.parse(scipy.__version__) >= version.parse("1.7")

    def test_opencv_available(self) -> None:
        """OpenCV should be available."""
        import cv2

        assert version.parse(cv2.__version__) >= version.parse("4.5")

    def test_scikit_image_available(self) -> None:
        """scikit-image should be available."""
        import skimage

        assert version.parse(skimage.__version__) >= version.parse("0.19")

    def test_scipy_fftpack_dct(self) -> None:
        """SciPy fftpack DCT should be available."""
        from scipy.fftpack import dct

        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = dct(x)
        assert result is not None


class TestFixtures:
    """Tests for pytest fixtures."""

    def test_sample_grayscale_fixture(self, sample_grayscale: np.ndarray) -> None:
        """sample_grayscale fixture should return correct shape and dtype."""
        assert sample_grayscale.shape == (128, 128)
        assert sample_grayscale.dtype == np.uint8

    def test_sample_color_fixture(self, sample_color: np.ndarray) -> None:
        """sample_color fixture should return correct shape and dtype."""
        assert sample_color.shape == (128, 128, 3)
        assert sample_color.dtype == np.uint8

    def test_low_contrast_fixture(self, low_contrast_image: np.ndarray) -> None:
        """low_contrast_image fixture should have narrow range."""
        assert low_contrast_image.min() >= 100
        assert low_contrast_image.max() <= 150

    def test_single_color_fixture(self, single_color_image: np.ndarray) -> None:
        """single_color_image fixture should be uniform."""
        assert np.all(single_color_image == 128)

    def test_small_image_fixture(self, small_image: np.ndarray) -> None:
        """small_image fixture should be tiny."""
        assert small_image.shape == (8, 8)

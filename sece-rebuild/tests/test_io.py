"""Tests for image I/O utilities.

Tests cover:
- Format detection and loading
- Alpha channel handling
- Bit depth conversion
- Format selection for saving
- Edge cases and error handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sece.io import load_image, save_image, ensure_uint8, SUPPORTED_FORMATS
from sece.io.reader import (
    ImageLoadError,
    UnsupportedFormatError,
)


class TestLoadImage:
    """Tests for load_image function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_grayscale(self):
        """Standard 64x64 grayscale test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    @pytest.fixture
    def sample_color(self):
        """Standard 64x64 color test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_rgba(self):
        """Standard 64x64 RGBA test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)

    def test_load_png_grayscale(self, temp_dir, sample_grayscale):
        """Load PNG grayscale image."""
        path = temp_dir / "test.png"
        import cv2

        cv2.imwrite(str(path), sample_grayscale)

        result = load_image(path)
        assert result.shape == sample_grayscale.shape
        np.testing.assert_array_equal(result, sample_grayscale)

    def test_load_png_color(self, temp_dir, sample_color):
        """Load PNG color image."""
        path = temp_dir / "test.png"
        import cv2

        cv2.imwrite(str(path), sample_color)

        result = load_image(path)
        assert result.shape == sample_color.shape
        np.testing.assert_array_equal(result, sample_color)

    def test_load_jpeg(self, temp_dir, sample_color):
        """Load JPEG image."""
        path = temp_dir / "test.jpg"
        import cv2

        cv2.imwrite(str(path), sample_color)

        result = load_image(path)
        assert result.shape == sample_color.shape
        # JPEG is lossy, so we check shape and dtype, not exact values
        assert result.dtype == np.uint8

    def test_load_tiff(self, temp_dir, sample_grayscale):
        """Load TIFF image."""
        path = temp_dir / "test.tif"
        import cv2

        cv2.imwrite(str(path), sample_grayscale)

        result = load_image(path)
        np.testing.assert_array_equal(result, sample_grayscale)

    def test_load_bmp(self, temp_dir, sample_color):
        """Load BMP image."""
        path = temp_dir / "test.bmp"
        import cv2

        cv2.imwrite(str(path), sample_color)

        result = load_image(path)
        np.testing.assert_array_equal(result, sample_color)

    def test_load_webp(self, temp_dir, sample_color):
        """Load WebP image."""
        path = temp_dir / "test.webp"
        import cv2

        cv2.imwrite(str(path), sample_color)

        result = load_image(path)
        assert result.shape == sample_color.shape
        assert result.dtype == np.uint8

    def test_format_detection_from_extension(self, temp_dir, sample_color):
        """Format is correctly detected from file extension."""
        extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]
        import cv2

        for ext in extensions:
            path = temp_dir / f"test{ext}"
            cv2.imwrite(str(path), sample_color)
            result = load_image(path)
            assert result is not None, f"Failed to load {ext}"

    def test_rgba_converted_to_rgb_with_warning(self, temp_dir, sample_rgba):
        """RGBA images are converted to RGB with UserWarning."""
        path = temp_dir / "rgba.png"
        import cv2

        # Create RGBA image with IMWRITE_PNG_COMPRESSION
        cv2.imwrite(str(path), sample_rgba)

        with pytest.warns(UserWarning, match="Alpha"):
            result = load_image(path)

        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB, not RGBA

    def test_unsupported_format_raises_error(self, temp_dir):
        """Unsupported format raises UnsupportedFormatError."""
        path = temp_dir / "test.xyz"
        path.write_bytes(b"not an image")

        with pytest.raises(UnsupportedFormatError, match="Unsupported"):
            load_image(path)

    def test_missing_file_raises_error(self, temp_dir):
        """Missing file raises ImageLoadError."""
        path = temp_dir / "nonexistent.png"

        with pytest.raises(ImageLoadError, match="Failed to load"):
            load_image(path)

    def test_pathlib_path_works(self, temp_dir, sample_grayscale):
        """Path objects work as input."""
        path = temp_dir / "test.png"
        import cv2

        cv2.imwrite(str(path), sample_grayscale)

        result = load_image(path)  # Pass Path object directly
        assert result is not None

    def test_color_mode_grayscale(self, temp_dir, sample_color):
        """color_mode='grayscale' forces grayscale output."""
        path = temp_dir / "test.png"
        import cv2

        cv2.imwrite(str(path), sample_color)

        result = load_image(path, color_mode="grayscale")
        assert result.ndim == 2  # Grayscale

    def test_color_mode_color(self, temp_dir, sample_grayscale):
        """color_mode='color' forces color output."""
        path = temp_dir / "test.png"
        import cv2

        cv2.imwrite(str(path), sample_grayscale)

        result = load_image(path, color_mode="color")
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_ensure_rgb_converts_grayscale(self, temp_dir, sample_grayscale):
        """ensure_rgb=True converts grayscale to 3-channel."""
        path = temp_dir / "test.png"
        import cv2

        cv2.imwrite(str(path), sample_grayscale)

        result = load_image(path, ensure_rgb=True)
        assert result.ndim == 3
        assert result.shape[2] == 3


class TestEnsureUint8:
    """Tests for ensure_uint8 function."""

    def test_uint8_unchanged(self):
        """uint8 images are returned unchanged."""
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = ensure_uint8(image)
        np.testing.assert_array_equal(result, image)

    def test_uint16_scaled(self):
        """uint16 images are scaled to uint8 with warning."""
        image = np.random.randint(0, 65536, (64, 64), dtype=np.uint16)
        with pytest.warns(UserWarning, match="uint16"):
            result = ensure_uint8(image)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_float_0_1_scaled(self):
        """float images in [0, 1] are scaled to [0, 255]."""
        image = np.random.rand(64, 64).astype(np.float32)
        with pytest.warns(UserWarning, match="float32"):
            result = ensure_uint8(image)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_float_0_255_converted(self):
        """float images in [0, 255] are cast to uint8."""
        image = np.random.randint(0, 256, (64, 64)).astype(np.float64)
        with pytest.warns(UserWarning, match="float64"):
            result = ensure_uint8(image)
        assert result.dtype == np.uint8

    def test_int32_scaled(self):
        """int32 images are scaled to uint8."""
        image = np.random.randint(0, 1000, (64, 64), dtype=np.int32)
        with pytest.warns(UserWarning, match="int32"):
            result = ensure_uint8(image)
        assert result.dtype == np.uint8

    def test_constant_image_int32(self):
        """Constant int32 image returns zeros."""
        image = np.full((64, 64), 100, dtype=np.int32)
        with pytest.warns(UserWarning):
            result = ensure_uint8(image)
        assert result.dtype == np.uint8

    def test_unsupported_dtype_raises(self):
        """Unsupported dtype raises ValueError."""
        image = np.array(["a", "b", "c"])
        with pytest.raises(ValueError, match="Unsupported dtype"):
            ensure_uint8(image)


class TestSaveImage:
    """Tests for save_image function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_grayscale(self):
        """Standard 64x64 grayscale test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    @pytest.fixture
    def sample_color(self):
        """Standard 64x64 color test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_save_png(self, temp_dir, sample_color):
        """Save PNG image."""
        path = temp_dir / "output.png"
        result = save_image(sample_color, path)

        assert result == path
        assert path.exists()

        # Verify content
        loaded = load_image(path)
        np.testing.assert_array_equal(loaded, sample_color)

    def test_save_jpeg(self, temp_dir, sample_color):
        """Save JPEG image."""
        path = temp_dir / "output.jpg"
        result = save_image(sample_color, path)

        assert result == path
        assert path.exists()

    def test_save_tiff(self, temp_dir, sample_grayscale):
        """Save TIFF image."""
        path = temp_dir / "output.tif"
        result = save_image(sample_grayscale, path)

        assert result == path
        assert path.exists()

        loaded = load_image(path)
        np.testing.assert_array_equal(loaded, sample_grayscale)

    def test_save_bmp(self, temp_dir, sample_color):
        """Save BMP image."""
        path = temp_dir / "output.bmp"
        result = save_image(sample_color, path)

        assert result == path
        assert path.exists()

    def test_format_parameter_overrides_extension(self, temp_dir, sample_color):
        """format parameter overrides path extension."""
        path = temp_dir / "output.xyz"
        result = save_image(sample_color, path, format="png")

        # Should save as PNG with .png extension
        # The function modifies the path to use the correct extension
        assert result.suffix == ".png"
        assert result.exists()

        # Verify it can be loaded as PNG
        loaded = load_image(result)
        np.testing.assert_array_equal(loaded, sample_color)

    def test_jpeg_quality_parameter(self, temp_dir, sample_color):
        """JPEG quality parameter affects output."""
        import cv2

        path_low = temp_dir / "low.jpg"
        path_high = temp_dir / "high.jpg"

        save_image(sample_color, path_low, quality=10)
        save_image(sample_color, path_high, quality=100)

        # Low quality should result in smaller file
        size_low = path_low.stat().st_size
        size_high = path_high.stat().st_size

        assert size_low < size_high

    def test_png_compression_parameter(self, temp_dir, sample_color):
        """PNG compression parameter affects output."""
        import cv2

        path_low = temp_dir / "low.png"
        path_high = temp_dir / "high.png"

        save_image(sample_color, path_low, compression=0)
        save_image(sample_color, path_high, compression=9)

        # High compression should result in smaller or equal file
        size_low = path_low.stat().st_size
        size_high = path_high.stat().st_size

        assert size_high <= size_low

    def test_creates_parent_directories(self, temp_dir, sample_color):
        """Parent directories are created if needed."""
        path = temp_dir / "subdir" / "nested" / "output.png"
        result = save_image(sample_color, path)

        assert result == path
        assert path.exists()
        assert path.parent.exists()

    def test_unsupported_format_raises_error(self, temp_dir, sample_color):
        """Unsupported format raises UnsupportedFormatError."""
        path = temp_dir / "output.xyz"

        with pytest.raises(UnsupportedFormatError, match="Unsupported"):
            save_image(sample_color, path)


class TestSupportedFormats:
    """Tests for SUPPORTED_FORMATS constant."""

    def test_contains_expected_formats(self):
        """SUPPORTED_FORMATS contains expected image formats."""
        expected = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        assert expected <= set(SUPPORTED_FORMATS.keys())

    def test_all_values_are_int(self):
        """All format flags are integers."""
        for ext, flag in SUPPORTED_FORMATS.items():
            assert isinstance(flag, int)


class TestRoundTrip:
    """Tests for load/save round trips."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_color(self):
        """Standard 64x64 color test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_png_roundtrip_exact(self, temp_dir, sample_color):
        """PNG roundtrip preserves image exactly (lossless)."""
        path = temp_dir / "roundtrip.png"
        save_image(sample_color, path)
        loaded = load_image(path)

        np.testing.assert_array_equal(loaded, sample_color)

    def test_tiff_roundtrip_exact(self, temp_dir, sample_color):
        """TIFF roundtrip preserves image exactly (lossless)."""
        path = temp_dir / "roundtrip.tif"
        save_image(sample_color, path)
        loaded = load_image(path)

        np.testing.assert_array_equal(loaded, sample_color)

    def test_bmp_roundtrip_exact(self, temp_dir, sample_color):
        """BMP roundtrip preserves image exactly (lossless)."""
        path = temp_dir / "roundtrip.bmp"
        save_image(sample_color, path)
        loaded = load_image(path)

        np.testing.assert_array_equal(loaded, sample_color)


class TestEdgeCases:
    """Edge case tests for I/O functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_small_image(self, temp_dir):
        """Very small images load correctly."""
        import cv2

        image = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        path = temp_dir / "small.png"
        cv2.imwrite(str(path), image)

        loaded = load_image(path)
        np.testing.assert_array_equal(loaded, image)

    def test_large_image(self, temp_dir):
        """Large images load correctly."""
        import cv2

        image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        path = temp_dir / "large.png"
        cv2.imwrite(str(path), image)

        loaded = load_image(path)
        np.testing.assert_array_equal(loaded, image)

    def test_single_pixel_image(self, temp_dir):
        """Single pixel image loads correctly."""
        import cv2

        image = np.array([[128]], dtype=np.uint8)
        path = temp_dir / "single.png"
        cv2.imwrite(str(path), image)

        loaded = load_image(path)
        np.testing.assert_array_equal(loaded, image)

    def test_wide_image(self, temp_dir):
        """Wide (non-square) image loads correctly."""
        import cv2

        image = np.random.randint(0, 256, (64, 256), dtype=np.uint8)
        path = temp_dir / "wide.png"
        cv2.imwrite(str(path), image)

        loaded = load_image(path)
        assert loaded.shape == (64, 256)

    def test_tall_image(self, temp_dir):
        """Tall (non-square) image loads correctly."""
        import cv2

        image = np.random.randint(0, 256, (256, 64), dtype=np.uint8)
        path = temp_dir / "tall.png"
        cv2.imwrite(str(path), image)

        loaded = load_image(path)
        assert loaded.shape == (256, 64)

"""Image reader with format detection and preprocessing.

Provides load_image() for loading images with automatic format detection,
alpha channel stripping, and bit depth conversion.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import cv2
import numpy as np

# Supported image formats with OpenCV flags
SUPPORTED_FORMATS: dict[str, int] = {
    ".png": cv2.IMREAD_UNCHANGED,
    ".jpg": cv2.IMREAD_COLOR,
    ".jpeg": cv2.IMREAD_COLOR,
    ".tif": cv2.IMREAD_UNCHANGED,
    ".tiff": cv2.IMREAD_UNCHANGED,
    ".bmp": cv2.IMREAD_COLOR,
    ".webp": cv2.IMREAD_COLOR,
}

# Type alias for image arrays
ImageArray = np.ndarray


class ImageLoadError(Exception):
    """Raised when image loading fails."""

    pass


class UnsupportedFormatError(Exception):
    """Raised when image format is not supported."""

    pass


def load_image(
    path: Union[str, Path],
    *,
    color_mode: str = "auto",
    ensure_rgb: bool = False,
) -> ImageArray:
    """Load an image with automatic format detection.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the image file.
    color_mode : str, optional
        Color mode: "auto" (detect from file), "grayscale", "color".
        Default is "auto".
    ensure_rgb : bool, optional
        If True, convert grayscale to RGB (3-channel). Default is False.

    Returns
    -------
    np.ndarray
        Image array with shape (H, W) for grayscale or (H, W, 3) for color.
        dtype is uint8.

    Raises
    ------
    ImageLoadError
        If the image cannot be loaded.
    UnsupportedFormatError
        If the image format is not supported.

    Warns
    -----
    UserWarning
        If RGBA image is converted to RGB (alpha channel stripped).
    UserWarning
        If non-uint8 image is converted to 8-bit.
    """
    path = Path(path)

    # Check format support
    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported image format: {ext}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS.keys())}"
        )

    # Determine read flags
    flags = SUPPORTED_FORMATS[ext]
    if color_mode == "grayscale":
        flags = cv2.IMREAD_GRAYSCALE
    elif color_mode == "color":
        flags = cv2.IMREAD_COLOR

    # Load image
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ImageLoadError(f"Failed to load image: {path}")

    # Handle RGBA images
    image = _handle_alpha(image)

    # Ensure uint8
    image = ensure_uint8(image)

    # Convert grayscale to RGB if requested
    if ensure_rgb and image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def _handle_alpha(image: ImageArray) -> ImageArray:
    """Strip alpha channel from RGBA images with warning.

    Parameters
    ----------
    image : np.ndarray
        Input image array.

    Returns
    -------
    np.ndarray
        Image with alpha channel removed if present.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        warnings.warn(
            "Alpha channel detected and removed. "
            "Transparency information will be lost.",
            UserWarning,
            stacklevel=3,
        )
        return image[:, :, :3]  # BGR only
    return image


def ensure_uint8(image: ImageArray) -> ImageArray:
    """Convert image to 8-bit with appropriate scaling.

    Parameters
    ----------
    image : np.ndarray
        Input image array.

    Returns
    -------
    np.ndarray
        Image array with dtype uint8.

    Warns
    -----
    UserWarning
        If conversion from non-uint8 is performed.

    Raises
    ------
    ValueError
        If dtype is not supported for conversion.
    """
    if image.dtype == np.uint8:
        return image

    warnings.warn(
        f"Converting {image.dtype} to uint8",
        UserWarning,
        stacklevel=2,
    )

    if image.dtype == np.uint16:
        # Scale 16-bit to 8-bit
        return (image / 256).astype(np.uint8)

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            # Scale [0, 1] to [0, 255]
            return (image * 255).astype(np.uint8)
        else:
            # Assume already in [0, 255] range
            return image.astype(np.uint8)

    if image.dtype == np.int32 or image.dtype == np.int64:
        # Scale to 8-bit
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            return ((image - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
        return np.zeros_like(image, dtype=np.uint8)

    raise ValueError(f"Unsupported dtype for conversion: {image.dtype}")

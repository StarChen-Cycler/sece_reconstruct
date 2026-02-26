"""Image writer with format selection.

Provides save_image() for saving images with automatic format detection
from path extension or explicit format parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from sece.io.reader import SUPPORTED_FORMATS

# Type alias for image arrays
ImageArray = np.ndarray


class ImageSaveError(Exception):
    """Raised when image saving fails."""

    pass


def save_image(
    image: ImageArray,
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    quality: Optional[int] = None,
    compression: Optional[int] = None,
) -> Path:
    """Save an image with automatic format detection.

    Parameters
    ----------
    image : np.ndarray
        Image array to save. Should be uint8 with shape (H, W) for grayscale
        or (H, W, 3) for color (BGR format).
    path : Union[str, Path]
        Output path for the image.
    format : str, optional
        Explicit format override (e.g., "png", "jpg"). If None, format is
        detected from path extension.
    quality : int, optional
        Quality for JPEG (1-100, default 95). Ignored for other formats.
    compression : int, optional
        Compression level for PNG (0-9, default 3). Ignored for other formats.

    Returns
    -------
    Path
        The path where the image was saved.

    Raises
    ------
    ImageSaveError
        If the image cannot be saved.
    UnsupportedFormatError
        If the format is not supported.

    Notes
    -----
    - JPEG uses lossy compression; quality 95 is recommended for good quality.
    - PNG uses lossless compression; compression 3 balances speed and size.
    - TIFF and BMP are lossless.
    """
    path = Path(path)

    # Determine format
    if format:
        ext = f".{format.lower().lstrip('.')}"
    else:
        ext = path.suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        from sece.io.reader import UnsupportedFormatError

        raise UnsupportedFormatError(
            f"Unsupported image format: {ext}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS.keys())}"
        )

    # If format override is provided, modify the path extension
    # OpenCV uses file extension to determine format
    if format and path.suffix.lower() != ext:
        path = path.with_suffix(ext)

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Set format-specific parameters
    params: list[int] = []

    if ext in (".jpg", ".jpeg"):
        # JPEG quality
        params = [cv2.IMWRITE_JPEG_QUALITY, quality if quality else 95]
    elif ext == ".png":
        # PNG compression level
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression if compression else 3]
    elif ext in (".tif", ".tiff"):
        # TIFF compression
        params = [cv2.IMWRITE_TIFF_COMPRESSION, compression if compression else 1]

    # Save image
    success = cv2.imwrite(str(path), image, params)
    if not success:
        raise ImageSaveError(f"Failed to save image: {path}")

    return path

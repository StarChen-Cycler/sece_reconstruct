"""Color processing module for SECE/SECEDCT.

This module provides color space processors for luminance-based
contrast enhancement. The strategy pattern allows different color
spaces (HSV, LAB, YCbCr) to be used interchangeably.

Priority order (from image-processing-rules.md):
1. HSV (default): Perceptually uniform luminance (V channel)
2. LAB: Device-independent, good for scientific applications
3. YCbCr: Video standard, good for JPEG-like processing

Example
-------
>>> import cv2
>>> from sece.color import color_sece, color_secedct
>>> image = cv2.imread("photo.jpg")  # BGR format
>>> enhanced = color_sece(image)  # HSV by default
>>> enhanced_lab = color_sece(image, color_space="lab")
>>> enhanced_local = color_secedct(image, gamma=0.5)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from sece.color.processor import ColorProcessor
from sece.color.hsv import HSVProcessor
from sece.color.lab import LABProcessor
from sece.color.ycbcr import YCbCrProcessor
from sece.core import SECEResult, sece
from sece.secedct import SECEDCTResult, secedct

# Type alias for supported color spaces
ColorSpace = Literal["hsv", "lab", "ycbcr"]

__all__ = [
    # Processors
    "ColorProcessor",
    "HSVProcessor",
    "LABProcessor",
    "YCbCrProcessor",
    "get_processor",
    # Enhancement functions
    "color_sece",
    "color_sece_simple",
    "ColorSECEResult",
    "color_secedct",
    "color_secedct_simple",
    "ColorSECEDCTResult",
    # Type alias
    "ColorSpace",
]

# Registry of available processors
_PROCESSORS = {
    "hsv": HSVProcessor,
    "lab": LABProcessor,
    "ycbcr": YCbCrProcessor,
}


def get_processor(color_space: str) -> ColorProcessor:
    """Get color processor for the specified color space.

    Parameters
    ----------
    color_space : str
        Color space name ('hsv', 'lab', or 'ycbcr').

    Returns
    -------
    ColorProcessor
        Processor instance for the specified color space.

    Raises
    ------
    ValueError
        If color space is not supported.
    """
    color_space_lower = color_space.lower()
    if color_space_lower not in _PROCESSORS:
        available = ", ".join(sorted(_PROCESSORS.keys()))
        raise ValueError(
            f"Unsupported color space '{color_space}'. "
            f"Available: {available}"
        )
    return _PROCESSORS[color_space_lower]()


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class ColorSECEResult:
    """Result from color SECE enhancement.

    Attributes
    ----------
    image : NDArray[np.uint8]
        Enhanced BGR image with same shape as input.
    sece_result : SECEResult
        Intermediate SECE result from luminance processing.
    color_space : str
        Color space used for processing.
    processing_time_ms : float
        Total processing time in milliseconds.
    """

    image: NDArray[np.uint8]
    sece_result: SECEResult
    color_space: str
    processing_time_ms: float


@dataclass
class ColorSECEDCTResult:
    """Result from color SECEDCT enhancement.

    Attributes
    ----------
    image : NDArray[np.uint8]
        Enhanced BGR image with same shape as input.
    secedct_result : SECEDCTResult
        Intermediate SECEDCT result from luminance processing.
    color_space : str
        Color space used for processing.
    processing_time_ms : float
        Total processing time in milliseconds.
    """

    image: NDArray[np.uint8]
    secedct_result: SECEDCTResult
    color_space: str
    processing_time_ms: float


# =============================================================================
# Enhancement functions
# =============================================================================


def color_sece(
    image: NDArray[np.uint8],
    color_space: ColorSpace = "hsv",
    y_d: int = 0,
    y_u: int = 255,
) -> ColorSECEResult:
    """Apply SECE to color image.

    Converts the image to the specified color space, enhances the
    luminance channel using SECE, then converts back to BGR.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input BGR image of shape (H, W, 3).
    color_space : ColorSpace, optional
        Color space for processing ('hsv', 'lab', 'ycbcr'),
        by default 'hsv'.
    y_d : int, optional
        Lower bound of output range, by default 0.
    y_u : int, optional
        Upper bound of output range, by default 255.

    Returns
    -------
    ColorSECEResult
        Contains enhanced BGR image and intermediate results.

    Raises
    ------
    ValueError
        If image is not 3-channel uint8 BGR or color_space is invalid.

    Examples
    --------
    >>> import cv2
    >>> image = cv2.imread("photo.jpg")
    >>> result = color_sece(image)
    >>> result.image.shape == image.shape
    True
    >>> result.image.dtype == np.uint8
    True
    """
    start_time = time.perf_counter()

    # Get color processor
    processor = get_processor(color_space)
    processor.validate_input(image)

    # Extract luminance
    luminance, chrominance = processor.to_luminance(image)

    # Apply SECE to luminance
    sece_result = sece(luminance, y_d, y_u)

    # Combine enhanced luminance with chrominance
    enhanced_bgr = processor.from_luminance(sece_result.image, chrominance)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return ColorSECEResult(
        image=enhanced_bgr,
        sece_result=sece_result,
        color_space=processor.name,
        processing_time_ms=elapsed_ms,
    )


def color_secedct(
    image: NDArray[np.uint8],
    gamma: float = 0.5,
    color_space: ColorSpace = "hsv",
    y_d: int = 0,
    y_u: int = 255,
) -> ColorSECEDCTResult:
    """Apply SECEDCT to color image.

    Converts the image to the specified color space, enhances the
    luminance channel using SECEDCT (global SECE + local DCT), then
    converts back to BGR.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input BGR image of shape (H, W, 3).
    gamma : float, optional
        Local enhancement level in [0, 1], by default 0.5.
        - gamma=0: No local enhancement, output equals SECE result
        - gamma=1: Maximum local contrast enhancement
    color_space : ColorSpace, optional
        Color space for processing ('hsv', 'lab', 'ycbcr'),
        by default 'hsv'.
    y_d : int, optional
        Lower bound of output range for SECE stage, by default 0.
    y_u : int, optional
        Upper bound of output range for SECE stage, by default 255.

    Returns
    -------
    ColorSECEDCTResult
        Contains enhanced BGR image and intermediate results.

    Raises
    ------
    ValueError
        If image is not 3-channel uint8 BGR, gamma out of range,
        or color_space is invalid.

    Examples
    --------
    >>> import cv2
    >>> image = cv2.imread("photo.jpg")
    >>> result = color_secedct(image, gamma=0.5)
    >>> result.image.shape == image.shape
    True
    >>> result.image.dtype == np.uint8
    True
    """
    start_time = time.perf_counter()

    # Get color processor
    processor = get_processor(color_space)
    processor.validate_input(image)

    # Extract luminance
    luminance, chrominance = processor.to_luminance(image)

    # Apply SECEDCT to luminance
    secedct_result = secedct(luminance, gamma, y_d, y_u)

    # Combine enhanced luminance with chrominance
    enhanced_bgr = processor.from_luminance(secedct_result.image, chrominance)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return ColorSECEDCTResult(
        image=enhanced_bgr,
        secedct_result=secedct_result,
        color_space=processor.name,
        processing_time_ms=elapsed_ms,
    )


def color_sece_simple(
    image: NDArray[np.uint8],
    color_space: ColorSpace = "hsv",
    y_d: int = 0,
    y_u: int = 255,
) -> NDArray[np.uint8]:
    """Simplified color SECE that returns only the enhanced image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input BGR image of shape (H, W, 3).
    color_space : ColorSpace, optional
        Color space for processing, by default 'hsv'.
    y_d : int, optional
        Lower bound of output range, by default 0.
    y_u : int, optional
        Upper bound of output range, by default 255.

    Returns
    -------
    NDArray[np.uint8]
        Enhanced BGR image with same shape as input.
    """
    result = color_sece(image, color_space, y_d, y_u)
    return result.image


def color_secedct_simple(
    image: NDArray[np.uint8],
    gamma: float = 0.5,
    color_space: ColorSpace = "hsv",
    y_d: int = 0,
    y_u: int = 255,
) -> NDArray[np.uint8]:
    """Simplified color SECEDCT that returns only the enhanced image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input BGR image of shape (H, W, 3).
    gamma : float, optional
        Local enhancement level in [0, 1], by default 0.5.
    color_space : ColorSpace, optional
        Color space for processing, by default 'hsv'.
    y_d : int, optional
        Lower bound of output range, by default 0.
    y_u : int, optional
        Upper bound of output range, by default 255.

    Returns
    -------
    NDArray[np.uint8]
        Enhanced BGR image with same shape as input.
    """
    result = color_secedct(image, gamma, color_space, y_d, y_u)
    return result.image

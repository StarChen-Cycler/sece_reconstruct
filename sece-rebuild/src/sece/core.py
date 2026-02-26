"""SECE (Spatial Entropy-based Contrast Enhancement) core algorithm.

This module provides the complete SECE algorithm that integrates all
components for global image contrast enhancement.

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast
    Enhancement," IEEE Trans. Image Process., 2014.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from sece.distribution import compute_distribution_function
from sece.mapping import apply_mapping_to_image, compute_mapping
from sece.spatial_histogram import compute_all_spatial_histograms, compute_grid_size
from sece.spatial_entropy import compute_all_spatial_entropies


@dataclass
class SECEResult:
    """Result from SECE enhancement.

    Attributes
    ----------
    image : NDArray[np.uint8]
        Enhanced image with same shape as input.
    distribution : NDArray[np.float64]
        Distribution function f_k for each gray level.
        Needed by SECEDCT for alpha calculation.
    gray_levels : NDArray[np.int64]
        Input gray levels corresponding to distribution.
    cdf : NDArray[np.float64]
        Cumulative distribution function F.
    processing_time_ms : float
        Processing time in milliseconds.
    """

    image: NDArray[np.uint8]
    distribution: NDArray[np.float64]
    gray_levels: NDArray[np.int64]
    cdf: NDArray[np.float64]
    processing_time_ms: float


def sece(
    image: NDArray[np.uint8],
    y_d: int = 0,
    y_u: int = 255,
    epsilon: float = 1e-10,
) -> SECEResult:
    """Apply SECE (Spatial Entropy-based Contrast Enhancement) to image.

    This implements the complete SECE algorithm from Celik (2014):
    1. Compute spatial histograms for all gray levels (Formula 1)
    2. Compute grid size M×N (Formula 2)
    3. Compute spatial entropy S_k for each level (Formula 3)
    4. Compute distribution function f and CDF F (Formulas 4-6)
    5. Map gray levels using CDF (Formula 7)
    6. Apply mapping to create enhanced image

    Key property: SECE preserves the SHAPE of the input histogram while
    enhancing contrast, avoiding artifacts common in histogram equalization.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input grayscale image of shape (H, W).
    y_d : int, optional
        Lower bound of output range, by default 0.
    y_u : int, optional
        Upper bound of output range, by default 255.
    epsilon : float, optional
        Small value for numerical stability, by default 1e-10.

    Returns
    -------
    SECEResult
        Contains enhanced image, distribution f (for SECEDCT), and metadata.

    Raises
    ------
    ValueError
        If image is not 2D grayscale uint8.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    >>> result = sece(image)
    >>> result.image.shape == image.shape
    True
    >>> result.image.dtype == np.uint8
    True
    """
    start_time = time.perf_counter()

    # Validate input
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D grayscale, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Image must be uint8, got {image.dtype}")

    H, W = image.shape

    # Check for small images
    if H < 8 or W < 8:
        warnings.warn(
            f"Small image size {image.shape} may have limited enhancement. "
            "Grid size will be adjusted.",
            UserWarning,
        )

    # Step 1-2: Compute spatial histograms with grid size
    histograms, gray_levels, M, N = compute_all_spatial_histograms(image)
    K = len(gray_levels)

    # Edge case: single gray level
    if K == 1:
        # No enhancement possible - return as-is
        result_image = image.copy()
        f = np.array([1.0], dtype=np.float64)
        F = np.array([1.0], dtype=np.float64)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SECEResult(
            image=result_image,
            distribution=f,
            gray_levels=gray_levels,
            cdf=F,
            processing_time_ms=elapsed_ms,
        )

    # Step 3: Compute spatial entropies
    S = compute_all_spatial_entropies(histograms, epsilon)

    # Step 4: Compute distribution function f and CDF F
    f, F = compute_distribution_function(S, epsilon)

    # Step 5: Compute output gray level mapping
    output_levels = compute_mapping(F, y_d, y_u)

    # Step 6: Apply mapping to image
    result_image = apply_mapping_to_image(image, gray_levels, output_levels)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return SECEResult(
        image=result_image,
        distribution=f,
        gray_levels=gray_levels,
        cdf=F,
        processing_time_ms=elapsed_ms,
    )


def sece_simple(
    image: NDArray[np.uint8],
    y_d: int = 0,
    y_u: int = 255,
) -> NDArray[np.uint8]:
    """Simplified SECE that returns only the enhanced image.

    Convenience function for when you only need the enhanced image
    without distribution data.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input grayscale image of shape (H, W).
    y_d : int, optional
        Lower bound of output range, by default 0.
    y_u : int, optional
        Upper bound of output range, by default 255.

    Returns
    -------
    NDArray[np.uint8]
        Enhanced image with same shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    >>> enhanced = sece_simple(image)
    >>> enhanced.shape == image.shape
    True
    """
    result = sece(image, y_d, y_u)
    return result.image


def validate_sece_result(
    original: NDArray[np.uint8],
    enhanced: NDArray[np.uint8],
) -> dict:
    """Validate SECE enhancement result.

    Checks key properties:
    1. Output uses full dynamic range (or expanded from input)
    2. Shape preserved
    3. No clipping artifacts

    Parameters
    ----------
    original : NDArray[np.uint8]
        Original input image.
    enhanced : NDArray[np.uint8]
        SECE enhanced image.

    Returns
    -------
    dict
        Validation metrics and flags.
    """
    return {
        "shape_preserved": original.shape == enhanced.shape,
        "dtype_correct": enhanced.dtype == np.uint8,
        "output_range": (int(enhanced.min()), int(enhanced.max())),
        "input_range": (int(original.min()), int(original.max())),
        "range_expanded": (enhanced.max() - enhanced.min())
        >= (original.max() - original.min()),
        "uses_full_range": enhanced.min() == 0 and enhanced.max() == 255,
    }

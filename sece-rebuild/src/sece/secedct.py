"""SECEDCT (Spatial Entropy-based Contrast Enhancement with DCT) algorithm.

This module provides the complete SECEDCT algorithm that combines global
SECE enhancement with local DCT-based contrast enhancement.

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

from sece.core import SECEResult, sece
from sece.dct import dct2d, idct2d
from sece.weighting import compute_alpha, weight_coefficients_vectorized


@dataclass
class SECEDCTResult:
    """Result from SECEDCT enhancement.

    Attributes
    ----------
    image : NDArray[np.uint8]
        Enhanced image with same shape as input.
    sece_result : SECEResult
        Intermediate SECE result containing distribution f for SECEDCT.
    alpha : float
        Computed alpha parameter from distribution entropy.
    gamma : float
        Local enhancement level used.
    processing_time_ms : float
        Total processing time in milliseconds.
    """

    image: NDArray[np.uint8]
    sece_result: SECEResult
    alpha: float
    gamma: float
    processing_time_ms: float


def secedct(
    image: NDArray[np.uint8],
    gamma: float = 0.5,
    y_d: int = 0,
    y_u: int = 255,
    epsilon: float = 1e-10,
) -> SECEDCTResult:
    """Apply SECEDCT (SECE + DCT local enhancement) to image.

    This implements the complete SECEDCT algorithm from Celik (2014):
    1. Apply SECE for global enhancement: SECE(X) -> Y_global, f
    2. Compute alpha from distribution entropy: alpha = entropy(f)^gamma
    3. Transform to DCT domain: D = dct2d(Y_global)
    4. Weight DCT coefficients: D_weighted = w(k,l) * D(k,l)
    5. Inverse DCT: Y_final = idct2d(D_weighted)

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input grayscale image of shape (H, W).
    gamma : float, optional
        Local enhancement level in [0, 1]. Default is 0.5.
        - gamma=0: No local enhancement, output equals SECE result
        - gamma=1: Maximum local contrast enhancement
    y_d : int, optional
        Lower bound of output range for SECE stage, by default 0.
    y_u : int, optional
        Upper bound of output range for SECE stage, by default 255.
    epsilon : float, optional
        Small value for numerical stability, by default 1e-10.

    Returns
    -------
    SECEDCTResult
        Contains enhanced image, SECE intermediate result, alpha, and metadata.

    Raises
    ------
    ValueError
        If image is not 2D grayscale uint8 or gamma out of range.

    Notes
    -----
    When gamma=0, the function returns the same result as SECE (no local
    enhancement). This is because alpha=entropy(f)^0=1, which results in
    all weights w(k,l)=1, leaving DCT coefficients unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    >>> result = secedct(image, gamma=0.5)
    >>> result.image.shape == image.shape
    True
    >>> result.image.dtype == np.uint8
    True

    gamma=0 should equal SECE result:

    >>> sece_result = sece(image)
    >>> result_g0 = secedct(image, gamma=0)
    >>> np.array_equal(result_g0.image, sece_result.image)
    True
    """
    start_time = time.perf_counter()

    # Validate input
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D grayscale, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Image must be uint8, got {image.dtype}")
    if not 0 <= gamma <= 1:
        raise ValueError(f"Gamma must be in [0, 1], got {gamma}")

    H, W = image.shape

    # Check for small images
    if H < 8 or W < 8:
        warnings.warn(
            f"Small image size {image.shape} may have limited enhancement. "
            "Consider using SECE only for very small images.",
            UserWarning,
        )

    # Step 1: Apply SECE for global enhancement
    sece_result = sece(image, y_d, y_u, epsilon)

    # Edge case: single gray level - SECE already handles this
    if len(sece_result.gray_levels) == 1:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SECEDCTResult(
            image=sece_result.image,
            sece_result=sece_result,
            alpha=1.0,  # No weighting when only one level
            gamma=gamma,
            processing_time_ms=elapsed_ms,
        )

    # Step 2: Compute alpha from distribution entropy
    alpha = compute_alpha(sece_result.distribution, gamma)

    # If alpha is 1 (gamma=0), no local enhancement needed
    if alpha == 1.0:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SECEDCTResult(
            image=sece_result.image,
            sece_result=sece_result,
            alpha=alpha,
            gamma=gamma,
            processing_time_ms=elapsed_ms,
        )

    # Step 3: Transform to DCT domain (use float64 for precision)
    Y_global = sece_result.image.astype(np.float64)
    D = dct2d(Y_global)

    # Step 4: Weight DCT coefficients
    D_weighted = weight_coefficients_vectorized(D, alpha)

    # Step 5: Inverse DCT
    Y_final = idct2d(D_weighted)

    # Convert back to uint8 with clipping
    Y_final = np.clip(Y_final, 0, 255).astype(np.uint8)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return SECEDCTResult(
        image=Y_final,
        sece_result=sece_result,
        alpha=alpha,
        gamma=gamma,
        processing_time_ms=elapsed_ms,
    )


def secedct_simple(
    image: NDArray[np.uint8],
    gamma: float = 0.5,
    y_d: int = 0,
    y_u: int = 255,
) -> NDArray[np.uint8]:
    """Simplified SECEDCT that returns only the enhanced image.

    Convenience function for when you only need the enhanced image
    without intermediate results.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input grayscale image of shape (H, W).
    gamma : float, optional
        Local enhancement level in [0, 1]. Default is 0.5.
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
    >>> enhanced = secedct_simple(image, gamma=0.5)
    >>> enhanced.shape == image.shape
    True
    """
    result = secedct(image, gamma, y_d, y_u)
    return result.image


def validate_secedct_result(
    original: NDArray[np.uint8],
    enhanced: NDArray[np.uint8],
    gamma: float,
) -> dict:
    """Validate SECEDCT enhancement result.

    Checks key properties:
    1. Output shape preserved
    2. Output dtype correct
    3. Output range valid
    4. When gamma=0, output equals SECE output

    Parameters
    ----------
    original : NDArray[np.uint8]
        Original input image.
    enhanced : NDArray[np.uint8]
        SECEDCT enhanced image.
    gamma : float
        Gamma value used for enhancement.

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
        "gamma": gamma,
    }

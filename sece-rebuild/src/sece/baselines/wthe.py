"""Weighted Thresholded Histogram Equalization (WTHE) baseline.

Implementation of WTHE algorithm from the paper. This method applies
thresholding and weighting to the histogram before equalization to
prevent over-enhancement.

Reference:
    Q. Wang and R. K. Ward, "Fast Image/Video Contrast Enhancement Based
    on Weighted Thresholded Histogram Equalization," IEEE Trans. Consumer
    Electronics, 2007.
"""

from typing import TypeAlias

import numpy as np

# Type alias for image arrays
ImageArray: TypeAlias = np.ndarray

# Default WTHE parameters (from the paper)
DEFAULT_R = 0.5  # Threshold parameter
DEFAULT_V = 0.5  # Weight parameter


def wthe(
    image: ImageArray,
    r: float = DEFAULT_R,
    v: float = DEFAULT_V,
) -> np.ndarray:
    """
    Apply Weighted Thresholded Histogram Equalization (WTHE).

    WTHE improves upon standard histogram equalization by:
    1. Applying a threshold to the normalized histogram
    2. Weighting the histogram to control enhancement
    3. Equalizing the modified histogram

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (H, W) with dtype uint8.
    r : float, optional
        Threshold parameter for histogram clipping, by default 0.5.
        Range [0, 1]. Higher values preserve more of original histogram.
    v : float, optional
        Weight parameter for enhancement control, by default 0.5.
        Range [0, 1]. Higher values give more enhancement.

    Returns
    -------
    np.ndarray
        WTHE-enhanced image with same shape as input.

    Raises
    ------
    ValueError
        If input is not a 2D grayscale image or parameters out of range.

    Notes
    -----
    The WTHE algorithm:
    1. Compute normalized histogram h(i) for i in [0, 255]
    2. Compute threshold: T = r * max(h)
    3. Apply thresholding: h_t(i) = min(h(i), T)
    4. Apply weighting: h_w(i) = (h_t(i) / T)^v * T
    5. Normalize and compute CDF
    6. Apply mapping

    Parameters:
    - r: Controls the threshold level. r=1.0 gives standard HE.
    - v: Controls the weighting. v=1.0 gives thresholded HE without weighting.

    Examples
    --------
    >>> import numpy as np
    >>> from sece.baselines import wthe
    >>> # Low contrast image
    >>> low_contrast = np.random.randint(100, 150, (64, 64), dtype=np.uint8)
    >>> enhanced = wthe(low_contrast)
    >>> enhanced.shape == low_contrast.shape
    True
    """
    # Validate input
    if image.ndim != 2:
        raise ValueError(
            f"WTHE requires a 2D grayscale image. Got shape {image.shape}"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"WTHE requires uint8 image. Got dtype {image.dtype}"
        )

    if not 0 <= r <= 1:
        raise ValueError(f"Parameter r must be in [0, 1]. Got {r}")

    if not 0 <= v <= 1:
        raise ValueError(f"Parameter v must be in [0, 1]. Got {v}")

    # Compute histogram
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)

    # Normalize histogram
    total_pixels = image.size
    hist_norm = hist / total_pixels

    # Compute threshold
    threshold = r * np.max(hist_norm)

    # Apply thresholding (clip histogram)
    hist_thresholded = np.minimum(hist_norm, threshold)

    # Apply weighting
    # h_w(i) = (h_t(i) / T)^v * T for h_t(i) > 0, else 0
    hist_weighted = np.zeros_like(hist_thresholded)
    nonzero_mask = hist_thresholded > 0
    hist_weighted[nonzero_mask] = (
        (hist_thresholded[nonzero_mask] / threshold) ** v
    ) * threshold

    # Normalize the weighted histogram
    hist_sum = np.sum(hist_weighted)
    if hist_sum > 0:
        hist_weighted = hist_weighted / hist_sum

    # Compute CDF
    cdf = np.cumsum(hist_weighted)

    # Normalize CDF to [0, 255]
    cdf_normalized = (cdf * 255).astype(np.uint8)

    # Apply mapping
    result = cdf_normalized[image]

    return result


def wthe_with_params(
    image: ImageArray,
    r: float = DEFAULT_R,
    v: float = DEFAULT_V,
) -> dict:
    """
    Apply WTHE and return result with parameters used.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (H, W) with dtype uint8.
    r : float, optional
        Threshold parameter, by default 0.5.
    v : float, optional
        Weight parameter, by default 0.5.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'image': WTHE-enhanced image
        - 'r': Threshold parameter used
        - 'v': Weight parameter used

    Examples
    --------
    >>> from sece.baselines import wthe_with_params
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> result = wthe_with_params(img, r=0.7, v=0.6)
    >>> result['r']
    0.7
    """
    enhanced = wthe(image, r=r, v=v)

    return {
        "image": enhanced,
        "r": r,
        "v": v,
    }

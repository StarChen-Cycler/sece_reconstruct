"""GMSD (Gradient Magnitude Similarity Deviation) metric.

Implements the GMSD metric from Xue et al. (2014) for measuring perceptual
distortion between reference and distorted images. Uses Sobel gradients to
compute gradient magnitude similarity map, then takes standard deviation
as the final score. Lower GMSD indicates less distortion.

Reference:
    W. Xue, L. Zhang, X. Mou, and A. C. Bovik, "Gradient Magnitude
    Similarity Deviation: A Highly Efficient Perceptual Image Quality
    Index," IEEE Trans. Image Process., 2014.
"""

from typing import Optional, TypeAlias

import numpy as np

# Type alias for image arrays
ImageArray: TypeAlias = np.ndarray

# Constant from the original paper: c = (0.0026 * 255)^2 = 170
# This is used in the GMS computation for numerical stability
C_DEFAULT = 170.0  # (0.0026 * 255)^2 ≈ 170


def _compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude using Prewitt filters.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W) as float64

    Returns
    -------
    np.ndarray
        Gradient magnitude map (H, W)
    """
    # Prewitt operators for gradient computation (as used in original GMSD paper)
    # Horizontal gradient (dx)
    kernel_x = np.array(
        [[1/3, 0, -1/3],
         [1/3, 0, -1/3],
         [1/3, 0, -1/3]],
        dtype=np.float64
    )
    # Vertical gradient (dy)
    kernel_y = np.array(
        [[1/3, 1/3, 1/3],
         [0, 0, 0],
         [-1/3, -1/3, -1/3]],
        dtype=np.float64
    )

    H, W = image.shape

    # Pad image for convolution
    padded = np.pad(image, 1, mode='edge')

    # Compute gradients using convolution
    dx = np.zeros((H, W), dtype=np.float64)
    dy = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            window = padded[i:i+3, j:j+3]
            dx[i, j] = np.sum(window * kernel_x)
            dy[i, j] = np.sum(window * kernel_y)

    # Gradient magnitude
    gm = np.sqrt(dx**2 + dy**2)

    return gm


def _compute_gradient_magnitude_vectorized(image: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude using vectorized operations.

    Uses simple forward differences for efficiency, which approximates
    the Prewitt operators used in the original paper.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W) as float64

    Returns
    -------
    np.ndarray
        Gradient magnitude map (H, W)
    """
    # Compute gradients using forward differences
    # Horizontal: I(x+1, y) - I(x, y)
    dx = np.zeros_like(image)
    dx[:, :-1] = image[:, 1:] - image[:, :-1]

    # Vertical: I(x, y+1) - I(x, y)
    dy = np.zeros_like(image)
    dy[:-1, :] = image[1:, :] - image[:-1, :]

    # Gradient magnitude
    gm = np.sqrt(dx**2 + dy**2)

    return gm


def gmsd(
    reference: ImageArray,
    distorted: ImageArray,
    c: float = C_DEFAULT,
) -> float:
    """
    Compute Gradient Magnitude Similarity Deviation (GMSD).

    Computes the standard deviation of the Gradient Magnitude Similarity
    (GMS) map between reference and distorted images. Lower values indicate
    less distortion.

    Parameters
    ----------
    reference : np.ndarray
        Reference image (H, W) grayscale or (H, W, 3) color.
    distorted : np.ndarray
        Distorted/enhanced image, must have same shape as reference.
    c : float, optional
        Constant for numerical stability, by default 170.
        Original paper uses c = (0.0026 * 255)^2 ≈ 170.

    Returns
    -------
    float
        GMSD value in range [0, inf). Lower is better.
        GMSD = 0.0 for identical images.
        GMSD > 0.1 typically indicates visually noticeable distortion.

    Raises
    ------
    ValueError
        If images have different shapes.

    Notes
    -----
    The GMSD metric computes:
    1. Gradient magnitude maps for both images
    2. GMS map: GMS(x,y) = (2 * GM_r * GM_d + c) / (GM_r^2 + GM_d^2 + c)
    3. GMSD = std(GMS)

    GMSD captures local quality variations, making it sensitive to
    distortion distribution rather than just average quality.

    Examples
    --------
    >>> import numpy as np
    >>> from sece.metrics import gmsd
    >>> # Identical images have GMSD = 0.0
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> gmsd(img, img)
    0.0
    >>> # Different images have GMSD > 0
    >>> img2 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> gmsd(img, img2) > 0
    True
    """
    # Validate shapes match
    if reference.shape != distorted.shape:
        raise ValueError(
            f"Image shapes must match. Got {reference.shape} and {distorted.shape}"
        )

    # Convert to grayscale if needed
    if reference.ndim == 3:
        import cv2
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)

    # Convert to float64 for computation
    ref_float = reference.astype(np.float64)
    dis_float = distorted.astype(np.float64)

    # Compute gradient magnitudes
    gm_ref = _compute_gradient_magnitude_vectorized(ref_float)
    gm_dis = _compute_gradient_magnitude_vectorized(dis_float)

    # Compute Gradient Magnitude Similarity (GMS) map
    # GMS = (2 * GM_r * GM_d + c) / (GM_r^2 + GM_d^2 + c)
    gms_numerator = 2 * gm_ref * gm_dis + c
    gms_denominator = gm_ref**2 + gm_dis**2 + c
    gms_map = gms_numerator / gms_denominator

    # GMSD is the standard deviation of the GMS map
    gmsd_value = np.std(gms_map)

    return float(gmsd_value)


def gmsd_map(
    reference: ImageArray,
    distorted: ImageArray,
    c: float = C_DEFAULT,
) -> tuple[float, np.ndarray]:
    """
    Compute GMSD and return the full GMS map.

    Parameters
    ----------
    reference : np.ndarray
        Reference image (H, W) or (H, W, 3).
    distorted : np.ndarray
        Distorted image, must have same shape as reference.
    c : float, optional
        Constant for numerical stability, by default 170.

    Returns
    -------
    Tuple[float, np.ndarray]
        - GMSD value (standard deviation of GMS map)
        - Full GMS map (H, W) with values typically in [0, 1]

    Examples
    --------
    >>> import numpy as np
    >>> from sece.metrics import gmsd_map
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> value, gms_image = gmsd_map(img, img)
    >>> value
    0.0
    """
    # Validate shapes match
    if reference.shape != distorted.shape:
        raise ValueError(
            f"Image shapes must match. Got {reference.shape} and {distorted.shape}"
        )

    # Convert to grayscale if needed
    if reference.ndim == 3:
        import cv2
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)

    # Convert to float64 for computation
    ref_float = reference.astype(np.float64)
    dis_float = distorted.astype(np.float64)

    # Compute gradient magnitudes
    gm_ref = _compute_gradient_magnitude_vectorized(ref_float)
    gm_dis = _compute_gradient_magnitude_vectorized(dis_float)

    # Compute GMS map
    gms_numerator = 2 * gm_ref * gm_dis + c
    gms_denominator = gm_ref**2 + gm_dis**2 + c
    gms_map_result = gms_numerator / gms_denominator

    # GMSD is the standard deviation
    gmsd_value = np.std(gms_map_result)

    return float(gmsd_value), gms_map_result


def gmsd_comparison(
    reference: ImageArray,
    distorted: ImageArray,
    threshold: float = 0.1,
    c: float = C_DEFAULT,
) -> dict:
    """
    Compare images using GMSD and check for visually noticeable distortion.

    Parameters
    ----------
    reference : np.ndarray
        Reference image
    distorted : np.ndarray
        Distorted/enhanced image
    threshold : float, optional
        GMSD threshold for "visually noticeable distortion", by default 0.1
        GMSD > 0.1 typically indicates visible quality degradation.
    c : float, optional
        Constant for GMSD computation, by default 170.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'gmsd': GMSD value
        - 'visually_distorted': True if GMSD >= threshold
        - 'threshold': The threshold used

    Notes
    -----
    Threshold interpretation (from the paper):
    - GMSD < 0.05: Nearly imperceptible distortion
    - GMSD 0.05-0.1: Slightly perceptible
    - GMSD > 0.1: Visually noticeable distortion

    Examples
    --------
    >>> from sece import sece
    >>> from sece.metrics import gmsd_comparison
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> result = sece(img)
    >>> comparison = gmsd_comparison(img, result.image)
    >>> comparison['gmsd'] >= 0  # GMSD is non-negative
    True
    """
    gmsd_value = gmsd(reference, distorted, c=c)

    return {
        "gmsd": gmsd_value,
        "visually_distorted": gmsd_value >= threshold,
        "threshold": threshold,
    }

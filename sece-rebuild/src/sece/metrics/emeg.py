"""EMEG (Expected Measure of Enhancement by Gradient) metric.

Implements formula (14) from Celik (2014) for measuring contrast enhancement.
Divides image into blocks, computes gradient-based contrast measure for each block,
and averages results. Higher EMEG indicates higher contrast.

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast Enhancement,"
    IEEE Trans. Image Process., 2014.
"""

from typing import Union, TypeAlias

import numpy as np

# Type alias for image arrays
ImageArray: TypeAlias = np.ndarray


def emeg(
    image: ImageArray,
    block_size: int = 8,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Expected Measure of Enhancement by Gradient (EMEG).

    Formula (14) from the paper. Measures contrast enhancement level.
    Divides image into blocks, computes gradient ratio (max/min) for each
    block, applies log, and returns the average.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W) grayscale or (H, W, 3) color.
        Color images are converted to grayscale.
    block_size : int, optional
        Size of blocks for local contrast measurement, by default 8
    epsilon : float, optional
        Small constant to prevent log(0) and division by zero, by default 1e-10

    Returns
    -------
    float
        EMEG value. Higher values indicate higher contrast.
        Range is typically [0, 1] after normalization.

    Notes
    -----
    EMEG is sensitive to noise. A high EMEG may indicate noise rather
    than good contrast. For valid enhancement, EMEG(Y) > EMEG(X) is expected.

    The formula computes for each block k:
        EMEG_k = alpha * log(max(dx_h/dx_l, dy_h/dy_l))

    where dx_h, dx_l are max/min horizontal gradients in the block,
    and dy_h, dy_l are max/min vertical gradients.

    The final EMEG is normalized to [0, 1] range by dividing by a theoretical
    maximum (log(beta) where beta is the maximum possible gradient ratio).

    Examples
    --------
    >>> import numpy as np
    >>> from sece.metrics import emeg
    >>> # Black image has low EMEG
    >>> black = np.zeros((64, 64), dtype=np.uint8)
    >>> emeg(black) < 0.1
    True
    >>> # High contrast image has higher EMEG
    >>> high_contrast = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> emeg(high_contrast) > emeg(black)
    True
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        import cv2

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure float for gradient computation
    img_float = image.astype(np.float64)
    H, W = img_float.shape

    # Compute gradients using forward differences
    # Horizontal gradient: |I(x+1, y) - I(x, y)|
    dx = np.abs(np.diff(img_float, axis=1, prepend=img_float[:, 0:1]))
    # Vertical gradient: |I(x, y+1) - I(x, y)|
    dy = np.abs(np.diff(img_float, axis=0, prepend=img_float[0:1, :]))

    # Calculate number of blocks
    # Use floor division to avoid partial blocks
    k1 = H // block_size  # Number of blocks vertically
    k2 = W // block_size  # Number of blocks horizontally

    # Maximum gradient value for 8-bit images
    max_gradient = 255.0
    # Theoretical max ratio (max_gradient / epsilon) for normalization
    # Use log for normalization
    max_log_ratio = np.log(max_gradient / epsilon)

    if k1 == 0 or k2 == 0:
        # Image too small for block size, use whole image
        dx_h, dx_l = np.max(dx), np.min(dx)
        dy_h, dy_l = np.max(dy), np.min(dy)

        # Add epsilon to avoid log(0) when dx_l or dy_l is 0
        ratio_dx = (dx_h + epsilon) / (dx_l + epsilon)
        ratio_dy = (dy_h + epsilon) / (dy_l + epsilon)

        log_ratio = np.log(max(ratio_dx, ratio_dy))
        # Normalize to [0, 1]
        return max(0.0, min(1.0, log_ratio / max_log_ratio))

    total = 0.0
    count = 0

    for i in range(k1):
        for j in range(k2):
            # Extract block
            block_dx = dx[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            block_dy = dy[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]

            # Compute max/min gradients in block
            dx_h, dx_l = np.max(block_dx), np.min(block_dx)
            dy_h, dy_l = np.max(block_dy), np.min(block_dy)

            # Compute gradient ratios with epsilon protection
            ratio_dx = (dx_h + epsilon) / (dx_l + epsilon)
            ratio_dy = (dy_h + epsilon) / (dy_l + epsilon)

            # Apply log and add to total
            log_ratio = np.log(max(ratio_dx, ratio_dy))
            total += log_ratio
            count += 1

    # Average and normalize to [0, 1]
    avg_log_ratio = total / count if count > 0 else 0.0
    normalized = avg_log_ratio / max_log_ratio

    # Clamp to [0, 1]
    return max(0.0, min(1.0, normalized))


def emeg_comparison(
    original: ImageArray,
    enhanced: ImageArray,
    block_size: int = 8,
) -> dict:
    """
    Compare EMEG values between original and enhanced images.

    Parameters
    ----------
    original : np.ndarray
        Original image
    enhanced : np.ndarray
        Enhanced image
    block_size : int, optional
        Block size for EMEG computation, by default 8

    Returns
    -------
    dict
        Dictionary with keys:
        - 'original': EMEG of original image
        - 'enhanced': EMEG of enhanced image
        - 'improvement': enhanced - original
        - 'ratio': enhanced / original (or inf if original is 0)

    Examples
    --------
    >>> from sece import sece
    >>> from sece.metrics import emeg_comparison
    >>> import numpy as np
    >>> low_contrast = np.random.randint(100, 150, (64, 64), dtype=np.uint8)
    >>> result = sece(low_contrast)
    >>> comparison = emeg_comparison(low_contrast, result.image)
    >>> comparison['improvement'] > 0  # Enhancement should increase EMEG
    True
    """
    emeg_orig = emeg(original, block_size=block_size)
    emeg_enh = emeg(enhanced, block_size=block_size)

    improvement = emeg_enh - emeg_orig
    ratio = emeg_enh / emeg_orig if emeg_orig > 0 else float("inf")

    return {
        "original": emeg_orig,
        "enhanced": emeg_enh,
        "improvement": improvement,
        "ratio": ratio,
    }

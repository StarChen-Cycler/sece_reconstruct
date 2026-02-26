"""SSIM (Structural Similarity Index Measure) metric.

Wrapper around skimage.metrics.structural_similarity for computing
perceptual similarity between images.

Reference:
    Z. Wang et al., "Image quality assessment: From error visibility to
    structural similarity," IEEE TIP, 2004.
"""

from typing import Optional, Union, Tuple, TypeAlias

import numpy as np

# Type alias for image arrays
ImageArray: TypeAlias = np.ndarray


def ssim(
    image1: ImageArray,
    image2: ImageArray,
    win_size: Optional[int] = None,
    gaussian_weights: bool = True,
    data_range: Optional[float] = None,
    **kwargs,
) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM) between two images.

    Wrapper around skimage.metrics.structural_similarity.

    Parameters
    ----------
    image1 : np.ndarray
        First image (H, W) or (H, W, 3).
    image2 : np.ndarray
        Second image, must have same shape as image1.
    win_size : int, optional
        Side length of sliding window, by default 7 or 11.
        Must be odd.
    gaussian_weights : bool, optional
        Use Gaussian weighting for window, by default True.
    data_range : float, optional
        Value range of input images. If None, inferred from image dtype.
        For uint8 images, this is 255.
    **kwargs
        Additional arguments passed to structural_similarity.

    Returns
    -------
    float
        SSIM value in range [-1, 1]. Typically [0, 1] for natural images.
        SSIM = 1.0 for identical images.

    Raises
    ------
    ValueError
        If images have different shapes.

    Notes
    -----
    SSIM measures perceptual similarity by comparing:
    - Luminance (mean intensity)
    - Contrast (variance)
    - Structure (covariance normalized by variances)

    Threshold 0.9 is commonly used for "perceptually similar".

    Examples
    --------
    >>> import numpy as np
    >>> from sece.metrics import ssim
    >>> # Identical images have SSIM = 1.0
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> ssim(img, img)
    1.0
    >>> # Different images have SSIM < 1.0
    >>> img2 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> ssim(img, img2) < 1.0
    True
    """
    from skimage.metrics import structural_similarity

    # Validate shapes match
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image shapes must match. Got {image1.shape} and {image2.shape}"
        )

    # Determine if multichannel (color) image
    is_multichannel = image1.ndim == 3

    # Infer data_range if not provided
    if data_range is None:
        if image1.dtype == np.uint8:
            data_range = 255
        elif image1.dtype == np.uint16:
            data_range = 65535
        else:
            # For float images, use actual range
            data_range = max(image1.max(), image2.max()) - min(
                image1.min(), image2.min()
            )
            if data_range == 0:
                data_range = 1.0

    # Set default win_size if not provided
    if win_size is None:
        # Use smaller window for small images
        min_dim = min(image1.shape[:2])
        if min_dim < 7:
            win_size = 3
        elif min_dim < 16:
            win_size = 5
        elif min_dim < 64:
            win_size = 7
        else:
            win_size = 11

    # Ensure win_size is odd
    if win_size % 2 == 0:
        win_size += 1

    # Ensure win_size doesn't exceed image dimensions
    min_dim = min(image1.shape[:2])
    if win_size > min_dim:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        if win_size < 3:
            win_size = 3

    # Build kwargs for structural_similarity
    ssim_kwargs = {
        "win_size": win_size,
        "gaussian_weights": gaussian_weights,
        "data_range": data_range,
    }

    # Handle multichannel images - use channel_axis for newer skimage versions
    if is_multichannel:
        ssim_kwargs["channel_axis"] = 2  # Color channel is last axis

    # Add any additional kwargs (but don't override our settings)
    for key, value in kwargs.items():
        if key not in ssim_kwargs:
            ssim_kwargs[key] = value

    # Compute SSIM
    result = structural_similarity(image1, image2, **ssim_kwargs)

    return float(result)


def ssim_map(
    image1: ImageArray,
    image2: ImageArray,
    win_size: Optional[int] = None,
    gaussian_weights: bool = True,
    data_range: Optional[float] = None,
    **kwargs,
) -> Tuple[float, np.ndarray]:
    """
    Compute SSIM and return the full SSIM map.

    Parameters
    ----------
    image1 : np.ndarray
        First image (H, W) or (H, W, 3).
    image2 : np.ndarray
        Second image, must have same shape as image1.
    win_size : int, optional
        Side length of sliding window.
    gaussian_weights : bool, optional
        Use Gaussian weighting for window, by default True.
    data_range : float, optional
        Value range of input images.
    **kwargs
        Additional arguments passed to structural_similarity.

    Returns
    -------
    Tuple[float, np.ndarray]
        - Mean SSIM value
        - Full SSIM map (H', W') where H' < H, W' < W due to windowing

    Examples
    --------
    >>> import numpy as np
    >>> from sece.metrics import ssim_map
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> mean_ssim, ssim_image = ssim_map(img, img)
    >>> mean_ssim
    1.0
    """
    from skimage.metrics import structural_similarity

    # Validate shapes match
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image shapes must match. Got {image1.shape} and {image2.shape}"
        )

    # Determine if multichannel (color) image
    is_multichannel = image1.ndim == 3

    # Infer data_range if not provided
    if data_range is None:
        if image1.dtype == np.uint8:
            data_range = 255
        elif image1.dtype == np.uint16:
            data_range = 65535
        else:
            data_range = max(image1.max(), image2.max()) - min(
                image1.min(), image2.min()
            )
            if data_range == 0:
                data_range = 1.0

    # Set default win_size if not provided
    if win_size is None:
        min_dim = min(image1.shape[:2])
        if min_dim < 7:
            win_size = 3
        elif min_dim < 16:
            win_size = 5
        elif min_dim < 64:
            win_size = 7
        else:
            win_size = 11

    # Ensure win_size is odd
    if win_size % 2 == 0:
        win_size += 1

    # Ensure win_size doesn't exceed image dimensions
    min_dim = min(image1.shape[:2])
    if win_size > min_dim:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        if win_size < 3:
            win_size = 3

    # Build kwargs for structural_similarity
    ssim_kwargs = {
        "win_size": win_size,
        "gaussian_weights": gaussian_weights,
        "data_range": data_range,
        "full": True,
    }

    # Handle multichannel images
    if is_multichannel:
        ssim_kwargs["channel_axis"] = 2

    # Add any additional kwargs
    for key, value in kwargs.items():
        if key not in ssim_kwargs:
            ssim_kwargs[key] = value

    # Compute SSIM with full map
    mean_ssim, ssim_image = structural_similarity(image1, image2, **ssim_kwargs)

    return float(mean_ssim), ssim_image


def ssim_comparison(
    original: ImageArray,
    enhanced: ImageArray,
    threshold: float = 0.9,
) -> dict:
    """
    Compare images using SSIM and check if perceptually similar.

    Parameters
    ----------
    original : np.ndarray
        Original image
    enhanced : np.ndarray
        Enhanced image
    threshold : float, optional
        SSIM threshold for "perceptually similar", by default 0.9

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ssim': SSIM value
        - 'perceptually_similar': True if SSIM >= threshold
        - 'threshold': The threshold used

    Examples
    --------
    >>> from sece import sece
    >>> from sece.metrics import ssim_comparison
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> result = sece(img)
    >>> comparison = ssim_comparison(img, result.image)
    >>> comparison['ssim'] >= 0  # SSIM should be non-negative
    True
    """
    ssim_value = ssim(original, enhanced)

    return {
        "ssim": ssim_value,
        "perceptually_similar": ssim_value >= threshold,
        "threshold": threshold,
    }

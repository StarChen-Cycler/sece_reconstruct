"""Contrast Limited Adaptive Histogram Equalization (CLAHE) baseline.

Wrapper around OpenCV's cv2.createCLAHE() function. CLAHE improves upon
standard adaptive histogram equalization (AHE) by limiting contrast
amplification to reduce noise over-enhancement.

Reference:
    K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization,"
    in Graphics Gems IV, Academic Press, 1994.
"""

from typing import Optional, TypeAlias

import numpy as np

# Type alias for image arrays
ImageArray: TypeAlias = np.ndarray

# Default CLAHE parameters
DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID_SIZE = (8, 8)


def clahe(
    image: ImageArray,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE divides the image into small tiles and performs histogram
    equalization on each tile independently. Contrast limiting prevents
    noise over-amplification by clipping the histogram before equalization.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (H, W) with dtype uint8.
    clip_limit : float, optional
        Threshold for contrast limiting. Higher values give more contrast.
        Typical range [1.0, 4.0]. By default 2.0.
    tile_grid_size : tuple[int, int], optional
        Size of grid for histogram equalization. By default (8, 8).

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image with same shape as input.

    Raises
    ------
    ValueError
        If input is not a 2D grayscale image.

    Notes
    -----
    CLAHE parameters:
    - clip_limit: Controls contrast enhancement. Higher = more contrast but
      potentially more noise. Typical range [1.0, 4.0].
    - tile_grid_size: Size of contextual regions. Smaller tiles = more local
      enhancement but may introduce artifacts.

    CLAHE is particularly effective for:
    - Medical images (X-ray, CT, MRI)
    - Low-contrast images with gradual intensity changes
    - Images with varying lighting conditions

    Examples
    --------
    >>> import numpy as np
    >>> from sece.baselines import clahe
    >>> # Low contrast image
    >>> low_contrast = np.random.randint(100, 150, (64, 64), dtype=np.uint8)
    >>> enhanced = clahe(low_contrast)
    >>> enhanced.shape == low_contrast.shape
    True
    """
    import cv2

    # Validate input
    if image.ndim != 2:
        raise ValueError(
            f"CLAHE requires a 2D grayscale image. Got shape {image.shape}"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"CLAHE requires uint8 image. Got dtype {image.dtype}"
        )

    # Create CLAHE object
    clahe_obj = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    # Apply CLAHE
    result = clahe_obj.apply(image)

    return result


def clahe_with_params(
    image: ImageArray,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
) -> dict:
    """
    Apply CLAHE and return result with parameters used.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (H, W) with dtype uint8.
    clip_limit : float, optional
        Threshold for contrast limiting, by default 2.0.
    tile_grid_size : tuple[int, int], optional
        Size of grid for histogram equalization, by default (8, 8).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'image': CLAHE-enhanced image
        - 'clip_limit': Clip limit used
        - 'tile_grid_size': Tile grid size used

    Examples
    --------
    >>> from sece.baselines import clahe_with_params
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> result = clahe_with_params(img, clip_limit=3.0)
    >>> result['clip_limit']
    3.0
    """
    enhanced = clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    return {
        "image": enhanced,
        "clip_limit": clip_limit,
        "tile_grid_size": tile_grid_size,
    }

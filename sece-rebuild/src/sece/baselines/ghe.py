"""Global Histogram Equalization (GHE) baseline.

Wrapper around OpenCV's cv2.equalizeHist() function. This is the simplest
form of histogram equalization that globally equalizes the histogram of
a grayscale image.

Reference:
    R. C. Gonzalez and R. E. Woods, "Digital Image Processing,"
    3rd ed., Pearson, 2007.
"""

from typing import TypeAlias

import numpy as np

# Type alias for image arrays
ImageArray: TypeAlias = np.ndarray


def ghe(image: ImageArray) -> np.ndarray:
    """
    Apply Global Histogram Equalization to enhance image contrast.

    This is a wrapper around OpenCV's equalizeHist function. It globally
    equalizes the histogram of the input image, resulting in a uniformly
    distributed histogram.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (H, W) with dtype uint8.
        Color images are not supported and will raise an error.

    Returns
    -------
    np.ndarray
        Histogram-equalized image with same shape as input.
        Output uses full dynamic range [0, 255].

    Raises
    ------
    ValueError
        If input is not a 2D grayscale image.

    Notes
    -----
    GHE works by:
    1. Computing the histogram of the input image
    2. Computing the CDF of the histogram
    3. Mapping each intensity level using the normalized CDF

    This guarantees the output uses the full dynamic range [0, 255].

    GHE can over-enhance images with large uniform regions, causing
    noise amplification. CLAHE is often preferred for such images.

    Examples
    --------
    >>> import numpy as np
    >>> from sece.baselines import ghe
    >>> # Low contrast image
    >>> low_contrast = np.random.randint(100, 150, (64, 64), dtype=np.uint8)
    >>> enhanced = ghe(low_contrast)
    >>> # Output should use full dynamic range
    >>> enhanced.min() < 50 and enhanced.max() > 200
    True
    """
    import cv2

    # Validate input
    if image.ndim != 2:
        raise ValueError(
            f"GHE requires a 2D grayscale image. Got shape {image.shape}"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"GHE requires uint8 image. Got dtype {image.dtype}"
        )

    # Apply global histogram equalization
    result = cv2.equalizeHist(image)

    return result

"""Gray level mapping function for SECE algorithm.

Implements the mapping function that transforms input gray levels
to output gray levels using the cumulative distribution function.

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast
    Enhancement," IEEE Trans. Image Process., 2014.
    Formula (7)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_mapping(
    F: NDArray[np.float64],
    y_d: int = 0,
    y_u: int = 255,
) -> NDArray[np.uint8]:
    """Map input gray levels to output gray levels using CDF.

    Formula (7) from the paper:
        y_k = floor(F_k * (y_u - y_d) + y_d)

    This transforms the CDF values (in [0, 1]) to output gray levels
    in the range [y_d, y_u]. The mapping preserves the ordering
    (if F[i] < F[j], then y[i] <= y[j]).

    Parameters
    ----------
    F : NDArray[np.float64]
        Cumulative distribution function of shape (K,), where K is
        the number of distinct gray levels. Values should be in [0, 1].
    y_d : int, optional
        Lower bound of output range, by default 0.
    y_u : int, optional
        Upper bound of output range, by default 255.

    Returns
    -------
    NDArray[np.uint8]
        Output gray levels of shape (K,), with values in [y_d, y_u].

    Raises
    ------
    ValueError
        If F is empty, contains values outside [0, 1], or if y_d >= y_u.

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([0.1, 0.3, 0.6, 1.0])  # CDF for 4 gray levels
    >>> y = compute_mapping(F)
    >>> y
    array([ 25,  76, 153, 255], dtype=uint8)
    >>> y.min() >= 0 and y.max() <= 255
    True
    """
    F = np.asarray(F, dtype=np.float64)

    if F.size == 0:
        raise ValueError("CDF array F cannot be empty")

    if y_d >= y_u:
        raise ValueError(f"y_d ({y_d}) must be less than y_u ({y_u})")

    if np.any(F < 0) or np.any(F > 1):
        raise ValueError("CDF values must be in range [0, 1]")

    # Formula (7): y_k = floor(F_k * (y_u - y_d) + y_d)
    output_range = y_u - y_d
    y = np.floor(F * output_range + y_d)

    # Clip to ensure values are in valid range
    y = np.clip(y, y_d, y_u)

    # Convert to uint8 for image compatibility
    return y.astype(np.uint8)


def apply_mapping_to_image(
    image: NDArray[np.uint8],
    gray_levels: NDArray[np.int64],
    output_levels: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Apply gray level mapping to an image.

    For each pixel in the input image, look up its corresponding
    output gray level and create the mapped image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input grayscale image of shape (H, W).
    gray_levels : NDArray[np.int64]
        Input gray levels that correspond to the CDF/mapping.
        Shape (K,) where K is the number of distinct gray levels.
    output_levels : NDArray[np.uint8]
        Output gray levels from compute_mapping().
        Shape (K,) - must have same length as gray_levels.

    Returns
    -------
    NDArray[np.uint8]
        Mapped image of shape (H, W).

    Raises
    ------
    ValueError
        If gray_levels and output_levels have different lengths,
        or if image contains gray levels not in gray_levels.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    >>> gray_levels = np.array([0, 1, 2, 3])
    >>> output_levels = np.array([10, 60, 150, 255], dtype=np.uint8)
    >>> result = apply_mapping_to_image(image, gray_levels, output_levels)
    >>> result
    array([[ 10,  60],
           [150, 255]], dtype=uint8)
    """
    if len(gray_levels) != len(output_levels):
        raise ValueError(
            f"gray_levels ({len(gray_levels)}) and output_levels "
            f"({len(output_levels)}) must have same length"
        )

    # Create lookup table for all possible gray levels (0-255)
    lookup = np.zeros(256, dtype=np.uint8)

    # Map each input gray level to its output level
    for input_level, output_level in zip(gray_levels, output_levels):
        lookup[input_level] = output_level

    # Check that all gray levels in image are in our mapping
    unique_image_levels = np.unique(image)
    for level in unique_image_levels:
        if level not in gray_levels:
            raise ValueError(
                f"Image contains gray level {level} not in gray_levels"
            )

    # Apply lookup table to image
    return lookup[image]


def validate_mapping(
    output_levels: NDArray[np.uint8],
    y_d: int = 0,
    y_u: int = 255,
) -> bool:
    """Validate that output levels satisfy required properties.

    Properties checked:
    1. All values are in [y_d, y_u]
    2. Values are monotonically non-decreasing (since CDF is)
    3. First value >= y_d, last value <= y_u

    Parameters
    ----------
    output_levels : NDArray[np.uint8]
        Output gray levels from compute_mapping().
    y_d : int, optional
        Lower bound of output range, by default 0.
    y_u : int, optional
        Upper bound of output range, by default 255.

    Returns
    -------
    bool
        True if all properties are satisfied.

    Examples
    --------
    >>> import numpy as np
    >>> output_levels = np.array([25, 76, 153, 255], dtype=np.uint8)
    >>> validate_mapping(output_levels)
    True
    """
    # Convert to int64 to avoid uint8 underflow issues in diff
    output_int = output_levels.astype(np.int64)

    # Check bounds
    if np.any(output_int < y_d) or np.any(output_int > y_u):
        return False

    # Check monotonicity (non-decreasing)
    if not np.all(np.diff(output_int) >= 0):
        return False

    return True

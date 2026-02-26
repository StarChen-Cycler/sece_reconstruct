"""Spatial histogram computation for SECE algorithm.

Implements 2D spatial histogram computation where each gray level's
spatial distribution is captured in an M×N grid. This is the key
innovation of the SECE algorithm over traditional histogram equalization.

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast
    Enhancement," IEEE Trans. Image Process., 2014.
    Formulas (1) and (2)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def compute_grid_size(K: int, image_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Compute grid size M×N for spatial histogram.

    Formula (2) from the paper:
        N = floor(sqrt(K/r))
        M = floor(sqrt(K*r))
    where r = H/W is the image aspect ratio.

    The grid size is chosen such that the total number of grids M×N
    is approximately equal to K (number of distinct gray levels),
    while preserving the image aspect ratio.

    Parameters
    ----------
    K : int
        Number of distinct gray levels in the image.
    image_shape : Tuple[int, int]
        Image shape as (height, width).

    Returns
    -------
    Tuple[int, int]
        Grid dimensions (M, N) where M >= 1, N >= 1.

    Examples
    --------
    >>> compute_grid_size(100, (480, 640))
    (9, 12)
    >>> compute_grid_size(256, (512, 512))
    (16, 16)
    >>> compute_grid_size(10, (8, 8))  # Small image
    (3, 3)
    """
    H, W = image_shape

    # Aspect ratio r = H/W
    r = H / W

    # Formula (2): Grid size computation
    N = int(np.floor(np.sqrt(K / r)))
    M = int(np.floor(np.sqrt(K * r)))

    # Ensure minimum grid size of 1
    M = max(1, M)
    N = max(1, N)

    return M, N


def compute_spatial_histogram(
    image: NDArray[np.uint8],
    gray_level: int,
    M: int,
    N: int,
) -> NDArray[np.float64]:
    """Compute 2D spatial histogram for a specific gray level.

    Formula (1) from the paper:
        h_k[m, n] = count of pixels equal to x_k in grid cell (m, n)
        normalized by the cell area.

    The image is divided into M×N non-overlapping rectangular grids.
    For each grid cell, we count how many pixels have the specified
    gray level.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input grayscale image of shape (H, W).
    gray_level : int
        The gray level x_k to compute histogram for (0-255).
    M : int
        Number of grid rows.
    N : int
        Number of grid columns.

    Returns
    -------
    NDArray[np.float64]
        Normalized spatial histogram h_k of shape (M, N).
        Each element represents the proportion of pixels with
        gray_level in that grid cell.

    Raises
    ------
    ValueError
        If M or N is less than 1, or if image is not 2D.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    >>> h = compute_spatial_histogram(image, 1, 3, 3)
    >>> h.shape
    (3, 3)
    """
    if M < 1 or N < 1:
        raise ValueError(f"Grid dimensions must be >= 1, got M={M}, N={N}")

    if image.ndim != 2:
        raise ValueError(f"Image must be 2D grayscale, got shape {image.shape}")

    H, W = image.shape

    # Initialize histogram
    h_k = np.zeros((M, N), dtype=np.float64)

    # Compute grid cell boundaries
    # Each cell spans approximately H//M rows and W//N columns
    row_edges = np.linspace(0, H, M + 1, dtype=np.int64)
    col_edges = np.linspace(0, W, N + 1, dtype=np.int64)

    # Create mask for pixels with the target gray level
    mask = image == gray_level

    # Count pixels in each grid cell
    for m in range(M):
        row_start, row_end = row_edges[m], row_edges[m + 1]
        for n in range(N):
            col_start, col_end = col_edges[n], col_edges[n + 1]

            # Extract grid cell
            cell_mask = mask[row_start:row_end, col_start:col_end]

            # Count pixels with gray_level in this cell
            count = np.sum(cell_mask)

            # Normalize by cell area
            cell_area = (row_end - row_start) * (col_end - col_start)
            if cell_area > 0:
                h_k[m, n] = count / cell_area
            else:
                h_k[m, n] = 0.0

    return h_k


def compute_all_spatial_histograms(
    image: NDArray[np.uint8],
    M: int | None = None,
    N: int | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], int, int]:
    """Compute spatial histograms for all gray levels present in image.

    For efficiency, this computes histograms for all distinct gray levels
    in a single pass through the image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Input grayscale image of shape (H, W).
    M : int, optional
        Number of grid rows. If None, computed from K and image shape.
    N : int, optional
        Number of grid columns. If None, computed from K and image shape.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.int64], int, int]
        - histograms: Array of shape (K, M, N) containing spatial histograms
          for each gray level
        - gray_levels: Array of shape (K,) containing the distinct gray levels
        - M: Number of grid rows used
        - N: Number of grid columns used

    Examples
    --------
    >>> import numpy as np
    >>> image = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    >>> hists, levels, M, N = compute_all_spatial_histograms(image)
    >>> len(levels)  # Number of distinct gray levels
    2
    """
    H, W = image.shape

    # Get distinct gray levels
    gray_levels = np.unique(image)
    K = len(gray_levels)

    # Compute grid size if not provided
    if M is None or N is None:
        M, N = compute_grid_size(K, (H, W))
    else:
        if M < 1 or N < 1:
            raise ValueError(f"Grid dimensions must be >= 1, got M={M}, N={N}")

    # Initialize histograms array
    histograms = np.zeros((K, M, N), dtype=np.float64)

    # Compute grid cell boundaries
    row_edges = np.linspace(0, H, M + 1, dtype=np.int64)
    col_edges = np.linspace(0, W, N + 1, dtype=np.int64)

    # Pre-compute cell areas
    cell_areas = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        for n in range(N):
            cell_areas[m, n] = (
                (row_edges[m + 1] - row_edges[m])
                * (col_edges[n + 1] - col_edges[n])
            )

    # Build mapping from gray level to index
    level_to_idx = {level: idx for idx, level in enumerate(gray_levels)}

    # Count pixels in each grid cell for each gray level
    for m in range(M):
        row_start, row_end = row_edges[m], row_edges[m + 1]
        for n in range(N):
            col_start, col_end = col_edges[n], col_edges[n + 1]

            # Extract grid cell
            cell = image[row_start:row_end, col_start:col_end]

            # Count occurrences of each gray level in this cell
            for level in gray_levels:
                count = np.sum(cell == level)
                idx = level_to_idx[level]
                if cell_areas[m, n] > 0:
                    histograms[idx, m, n] = count / cell_areas[m, n]

    return histograms, gray_levels, M, N


def compute_spatial_entropy(h_k: NDArray[np.float64], epsilon: float = 1e-10) -> float:
    """Compute spatial entropy for a single gray level histogram.

    Formula (3) from the paper:
        S_k = -sum(h_k * log2(h_k))

    This measures the spatial dispersion of gray level x_k.
    Higher entropy means the gray level is more spread across the image.

    Parameters
    ----------
    h_k : NDArray[np.float64]
        Normalized spatial histogram of shape (M, N).
    epsilon : float, optional
        Small value to prevent log(0), by default 1e-10.

    Returns
    -------
    float
        Spatial entropy S_k >= 0.

    Examples
    --------
    >>> import numpy as np
    >>> h = np.array([[0.5, 0.5], [0.0, 0.0]])  # Concentrated in top row
    >>> entropy = compute_spatial_entropy(h)
    >>> entropy > 0
    True
    """
    # Flatten and filter out zeros
    h_flat = h_k.flatten()
    h_nonzero = h_flat[h_flat > epsilon]

    # Compute entropy: S_k = -sum(h_k * log2(h_k))
    if len(h_nonzero) == 0:
        return 0.0

    entropy = -np.sum(h_nonzero * np.log2(h_nonzero + epsilon))
    return max(0.0, float(entropy))


def compute_all_spatial_entropies(
    histograms: NDArray[np.float64],
    epsilon: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute spatial entropy for all gray level histograms.

    Parameters
    ----------
    histograms : NDArray[np.float64]
        Array of shape (K, M, N) containing spatial histograms.
    epsilon : float, optional
        Small value to prevent log(0), by default 1e-10.

    Returns
    -------
    NDArray[np.float64]
        Array of shape (K,) containing spatial entropy for each gray level.
    """
    K = histograms.shape[0]
    entropies = np.zeros(K, dtype=np.float64)

    for k in range(K):
        entropies[k] = compute_spatial_entropy(histograms[k], epsilon)

    return entropies

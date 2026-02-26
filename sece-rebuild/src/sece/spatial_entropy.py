"""Spatial entropy calculation for SECE algorithm.

Implements Shannon entropy calculation for 2D spatial histograms.
Higher entropy indicates gray levels that are more spread across
the image space.

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast
    Enhancement," IEEE Trans. Image Process., 2014.
    Formula (3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

# Re-export from spatial_histogram for backward compatibility
from sece.spatial_histogram import (
    compute_all_spatial_entropies as _compute_all_spatial_entropies,
    compute_spatial_entropy as _compute_spatial_entropy,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "compute_spatial_entropy",
    "compute_all_spatial_entropies",
]


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
    return _compute_spatial_entropy(h_k, epsilon)


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
    return _compute_all_spatial_entropies(histograms, epsilon)

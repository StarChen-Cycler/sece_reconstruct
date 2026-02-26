"""DCT coefficient weighting functions.

Implements formulas (10), (11), and (12) from the SECE paper for weighting
DCT coefficients in the SECEDCT algorithm.

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast
    Enhancement," IEEE Trans. Image Process., 2014.
"""

from __future__ import annotations

from typing import Union

import numpy as np

# Type alias for arrays
ArrayLike = Union[np.ndarray, list]


def compute_alpha(distribution: ArrayLike, gamma: float = 0.5) -> float:
    """Compute alpha parameter from distribution entropy.

    Implements formula (12) from the paper:
        alpha = entropy(f)^gamma

    The entropy is computed from the distribution function f (output from SECE).

    Parameters
    ----------
    distribution : array_like
        Distribution function f from SECE. Should sum to 1.0.
    gamma : float, optional
        Local enhancement level in [0, 1]. Default is 0.5.
        - gamma=0: alpha=1, no weighting (equivalent to SECE only)
        - gamma=1: alpha=entropy, maximum local enhancement

    Returns
    -------
    float
        Alpha value for DCT coefficient weighting.

    Notes
    -----
    Higher alpha increases weighting of high-frequency DCT coefficients,
    which enhances local contrast but may amplify noise.

    Examples
    --------
    >>> f = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    >>> alpha = compute_alpha(f, gamma=0.5)
    >>> alpha >= 1.0  # Entropy^0.5 is >= 1 for non-uniform distribution
    True
    """
    distribution = np.asarray(distribution, dtype=np.float64)

    # Normalize to ensure it's a proper distribution
    total = np.sum(distribution)
    if total > 0:
        distribution = distribution / total

    # Filter out zero probabilities for entropy calculation
    f_pos = distribution[distribution > 0]

    if len(f_pos) == 0:
        return 1.0

    # Compute Shannon entropy (base 2 as per paper)
    entropy = -np.sum(f_pos * np.log2(f_pos))

    # alpha = entropy^gamma
    alpha = entropy**gamma

    return float(alpha)


def weight_coefficients(
    D: ArrayLike,
    alpha: float,
) -> np.ndarray:
    """Weight DCT coefficients using formula (11).

    Implements formulas (10) and (11) from the paper:
        Y(k,l) = w(k,l) * D(k,l)
        w(k,l) = (1 + (alpha-1)*k/(H-1)) * (1 + (alpha-1)*l/(W-1))

    Parameters
    ----------
    D : array_like
        2D DCT coefficients from dct2d().
    alpha : float
        Weighting parameter from compute_alpha().

    Returns
    -------
    np.ndarray
        Weighted DCT coefficients with same shape as input.

    Notes
    -----
    - w(0,0) = 1 always, preserving the DC coefficient (mean value)
    - Higher-frequency coefficients are weighted more when alpha > 1
    - When alpha = 1, all weights are 1 (no change)

    Examples
    --------
    >>> D = np.random.randn(64, 64)
    >>> alpha = 2.0
    >>> D_weighted = weight_coefficients(D, alpha)
    >>> D_weighted[0, 0] == D[0, 0]  # DC coefficient unchanged
    True
    """
    D = np.asarray(D, dtype=np.float64)

    if D.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {D.shape}")

    H, W = D.shape

    # Handle edge case of 1x1 image
    if H == 1 and W == 1:
        return D.copy()

    # Create weight matrix
    weights = np.zeros((H, W), dtype=np.float64)

    for k in range(H):
        for l in range(W):
            # Formula (11): w(k,l)
            w_k = 1.0 + (alpha - 1) * k / max(H - 1, 1)
            w_l = 1.0 + (alpha - 1) * l / max(W - 1, 1)
            weights[k, l] = w_k * w_l

    # Apply weighting (formula 10)
    D_weighted = weights * D

    return D_weighted


def weight_coefficients_vectorized(
    D: ArrayLike,
    alpha: float,
) -> np.ndarray:
    """Vectorized version of weight_coefficients for better performance.

    Same functionality as weight_coefficients but using numpy broadcasting
    for faster computation on large arrays.

    Parameters
    ----------
    D : array_like
        2D DCT coefficients.
    alpha : float
        Weighting parameter.

    Returns
    -------
    np.ndarray
        Weighted DCT coefficients.
    """
    D = np.asarray(D, dtype=np.float64)

    if D.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {D.shape}")

    H, W = D.shape

    # Handle edge case
    if H == 1 and W == 1:
        return D.copy()

    # Create row and column indices
    k_indices = np.arange(H).reshape(-1, 1)  # Column vector
    l_indices = np.arange(W).reshape(1, -1)  # Row vector

    # Compute weights using broadcasting
    w_k = 1.0 + (alpha - 1) * k_indices / max(H - 1, 1)
    w_l = 1.0 + (alpha - 1) * l_indices / max(W - 1, 1)
    weights = w_k * w_l

    return weights * D


def compute_weight_matrix(H: int, W: int, alpha: float) -> np.ndarray:
    """Compute the weight matrix for given dimensions and alpha.

    Useful for visualization and debugging.

    Parameters
    ----------
    H : int
        Height (number of rows).
    W : int
        Width (number of columns).
    alpha : float
        Weighting parameter.

    Returns
    -------
    np.ndarray
        Weight matrix of shape (H, W).

    Examples
    --------
    >>> W = compute_weight_matrix(8, 8, alpha=2.0)
    >>> W[0, 0]  # DC weight is always 1
    1.0
    >>> W[7, 7]  # Highest frequency has highest weight
    4.0
    """
    if H == 1 and W == 1:
        return np.ones((1, 1), dtype=np.float64)

    k_indices = np.arange(H).reshape(-1, 1)
    l_indices = np.arange(W).reshape(1, -1)

    w_k = 1.0 + (alpha - 1) * k_indices / max(H - 1, 1)
    w_l = 1.0 + (alpha - 1) * l_indices / max(W - 1, 1)

    return w_k * w_l

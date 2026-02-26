"""Distribution function calculation for SECE algorithm.

Implements the discrete distribution function f_k and cumulative
distribution function F_k from spatial entropy values.

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast
    Enhancement," IEEE Trans. Image Process., 2014.
    Formulas (4), (5), (6)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def compute_distribution_function(
    S: NDArray[np.float64],
    epsilon: float = 1e-10,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute discrete distribution function f and CDF F from spatial entropies.

    This implements formulas (4), (5), and (6) from the paper:

    Formula (4): f_k = S_k / sum(S_l where l != k)
        - Measures relative importance of each gray level
        - A gray level with higher entropy is more spatially dispersed

    Formula (5): f_k = f_k / sum(f)
        - Normalization to ensure probability distribution

    Formula (6): F_k = cumulative sum of f_l for l = 1 to k
        - CDF for mapping function

    Parameters
    ----------
    S : NDArray[np.float64]
        Spatial entropy values of shape (K,) where K is the number of
        distinct gray levels.
    epsilon : float, optional
        Small value to prevent division by zero, by default 1e-10.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        - f: Normalized distribution function of shape (K,)
        - F: Cumulative distribution function of shape (K,)

    Raises
    ------
    ValueError
        If S is empty or contains negative values.

    Examples
    --------
    >>> import numpy as np
    >>> S = np.array([1.0, 2.0, 3.0])  # Three gray levels with entropies
    >>> f, F = compute_distribution_function(S)
    >>> f.sum()  # Should be 1.0 (normalized)
    1.0
    >>> F[-1]  # CDF should end at 1.0
    1.0
    >>> np.all(np.diff(F) >= 0)  # CDF should be monotonically increasing
    True
    """
    S = np.asarray(S, dtype=np.float64)

    if S.size == 0:
        raise ValueError("Spatial entropy array S cannot be empty")

    if np.any(S < 0):
        raise ValueError("Spatial entropy values must be non-negative")

    K = len(S)

    # Edge case: single gray level (K=1)
    # When there's only one gray level, it gets mapped to itself
    if K == 1:
        f = np.array([1.0], dtype=np.float64)
        F = np.array([1.0], dtype=np.float64)
        return f, F

    # Formula (4): f_k = S_k / sum(S_l where l != k)
    # Compute sum of all entropies except current one
    total_entropy = np.sum(S)

    # Initialize f
    f = np.zeros(K, dtype=np.float64)

    for k in range(K):
        # Sum of all entropies except S[k]
        sum_others = total_entropy - S[k]

        if sum_others > epsilon:
            f[k] = S[k] / sum_others
        else:
            # If all other entropies are zero, this level has relative importance 0
            f[k] = 0.0

    # Formula (5): Normalize to ensure sum(f) = 1
    f_sum = np.sum(f)

    if f_sum > epsilon:
        f = f / f_sum
    else:
        # Edge case: all f values are zero (shouldn't happen with valid S)
        # Fall back to uniform distribution
        f = np.ones(K, dtype=np.float64) / K

    # Formula (6): CDF F_k = cumulative sum of f_l
    F = np.cumsum(f)

    # Ensure F[-1] == 1.0 exactly (handle floating point precision)
    F = np.clip(F, 0.0, 1.0)

    return f, F


def validate_distribution(
    f: NDArray[np.float64],
    F: NDArray[np.float64],
    epsilon: float = 1e-10,
) -> bool:
    """Validate that distribution function satisfies required properties.

    Properties checked:
    1. sum(f) = 1 (probability distribution)
    2. F = cumsum(f) (CDF relationship)
    3. F is monotonically increasing
    4. F[0] >= 0 and F[-1] = 1

    Parameters
    ----------
    f : NDArray[np.float64]
        Distribution function.
    F : NDArray[np.float64]
        Cumulative distribution function.
    epsilon : float, optional
        Tolerance for floating point comparison, by default 1e-10.

    Returns
    -------
    bool
        True if all properties are satisfied.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.array([0.2, 0.3, 0.5])
    >>> F = np.cumsum(f)
    >>> validate_distribution(f, F)
    True
    """
    # Check sum(f) = 1
    if abs(np.sum(f) - 1.0) > epsilon:
        return False

    # Check F = cumsum(f)
    expected_F = np.cumsum(f)
    if not np.allclose(F, expected_F, atol=epsilon):
        return False

    # Check monotonically increasing
    if not np.all(np.diff(F) >= -epsilon):
        return False

    # Check bounds
    if F[0] < -epsilon or abs(F[-1] - 1.0) > epsilon:
        return False

    return True

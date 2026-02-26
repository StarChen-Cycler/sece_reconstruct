"""2D Discrete Cosine Transform functions.

Implements forward and inverse 2D-DCT using scipy.fftpack.
These functions implement formulas (8), (9), and (13) from the SECE paper.

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast
    Enhancement," IEEE Trans. Image Process., 2014.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.fftpack import dct, idct

# Type alias for arrays
ArrayLike = Union[np.ndarray, list]


def dct2d(x: ArrayLike) -> np.ndarray:
    """Compute 2D Discrete Cosine Transform.

    Implements formula (8) from the paper. The 2D-DCT is computed by
    applying 1D-DCT along rows, then along columns.

    Parameters
    ----------
    x : array_like
        Input 2D array (image or coefficients). Will be converted to
        float64 for computation.

    Returns
    -------
    np.ndarray
        2D-DCT coefficients with same shape as input.

    Notes
    -----
    Uses orthonormal DCT (norm='ortho') which ensures that:
    - Energy is preserved: sum(D^2) = sum(X^2)
    - idct2d(dct2d(x)) == x (within numerical precision)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(64, 64)
    >>> D = dct2d(x)
    >>> D.shape
    (64, 64)

    The inverse transform recovers the original:

    >>> recovered = idct2d(D)
    >>> np.allclose(x, recovered)
    True
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {x.shape}")

    # Apply 1D-DCT along columns (axis=0), then rows (axis=1)
    # Using orthonormal DCT for energy preservation
    result = dct(dct(x.T, norm="ortho").T, norm="ortho")

    return result


def idct2d(D: ArrayLike) -> np.ndarray:
    """Compute inverse 2D Discrete Cosine Transform.

    Implements formula (9) from the paper. The inverse 2D-DCT is computed
    by applying 1D-IDCT along rows, then along columns.

    Parameters
    ----------
    D : array_like
        Input 2D-DCT coefficients. Will be converted to float64.

    Returns
    -------
    np.ndarray
        Reconstructed spatial domain array with same shape as input.

    Notes
    -----
    Uses orthonormal IDCT (norm='ortho') which is the exact inverse
    of the orthonormal forward DCT.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(64, 64)
    >>> D = dct2d(x)
    >>> recovered = idct2d(D)
    >>> np.allclose(x, recovered)
    True
    """
    D = np.asarray(D, dtype=np.float64)

    if D.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {D.shape}")

    # Apply 1D-IDCT along columns (axis=0), then rows (axis=1)
    # Using orthonormal IDCT
    result = idct(idct(D.T, norm="ortho").T, norm="ortho")

    return result


def dct2d_blockwise(x: ArrayLike, block_size: int = 8) -> np.ndarray:
    """Compute blockwise 2D-DCT (used in JPEG-like compression).

    Divides image into non-overlapping blocks and computes 2D-DCT
    for each block independently.

    Parameters
    ----------
    x : array_like
        Input 2D array (image).
    block_size : int, optional
        Size of blocks (default 8, as used in JPEG).

    Returns
    -------
    np.ndarray
        Blockwise DCT coefficients with same shape as input.

    Notes
    -----
    This is provided for compatibility with JPEG-like processing,
    but is not used in the SECE/SECEDCT algorithms.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {x.shape}")

    H, W = x.shape

    # Pad to multiples of block_size if needed
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size

    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")

    H_new, W_new = x.shape
    result = np.zeros((H_new, W_new), dtype=np.float64)

    # Process each block
    for i in range(0, H_new, block_size):
        for j in range(0, W_new, block_size):
            block = x[i : i + block_size, j : j + block_size]
            result[i : i + block_size, j : j + block_size] = dct2d(block)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        result = result[:H, :W]

    return result


def idct2d_blockwise(D: ArrayLike, block_size: int = 8) -> np.ndarray:
    """Compute blockwise inverse 2D-DCT.

    Parameters
    ----------
    D : array_like
        Blockwise DCT coefficients.
    block_size : int, optional
        Size of blocks (default 8).

    Returns
    -------
    np.ndarray
        Reconstructed spatial domain array.
    """
    D = np.asarray(D, dtype=np.float64)

    if D.ndim != 2:
        raise ValueError(f"Input must be 2D array, got shape {D.shape}")

    H, W = D.shape

    # Pad to multiples of block_size if needed
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size

    if pad_h > 0 or pad_w > 0:
        D = np.pad(D, ((0, pad_h), (0, pad_w)), mode="reflect")

    H_new, W_new = D.shape
    result = np.zeros((H_new, W_new), dtype=np.float64)

    # Process each block
    for i in range(0, H_new, block_size):
        for j in range(0, W_new, block_size):
            block = D[i : i + block_size, j : j + block_size]
            result[i : i + block_size, j : j + block_size] = idct2d(block)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        result = result[:H, :W]

    return result

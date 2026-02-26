"""NumPy/SciPy backend for DCT transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.fftpack import dct, idct

from sece.backends.base import Backend

if TYPE_CHECKING:
    from typing import Union

    ArrayLike = Union[np.ndarray, list]


class NumpyBackend(Backend):
    """NumPy/SciPy backend for 2D-DCT transforms.

    Uses scipy.fftpack.dct with orthonormal normalization.
    CPU-only, but highly optimized for most image sizes.

    Examples
    --------
    >>> backend = NumpyBackend()
    >>> D = backend.dct2d(image)
    >>> recovered = backend.idct2d(D)
    >>> np.allclose(image, recovered)
    True
    """

    @property
    def name(self) -> str:
        """Backend name."""
        return "numpy"

    @property
    def device(self) -> str:
        """Device being used (always 'cpu' for NumPy)."""
        return "cpu"

    def dct2d(self, x: ArrayLike) -> np.ndarray:
        """Compute 2D Discrete Cosine Transform using SciPy.

        Parameters
        ----------
        x : array_like
            Input 2D array (image). Will be converted to float64.

        Returns
        -------
        np.ndarray
            2D-DCT coefficients with same shape as input.

        Raises
        ------
        ValueError
            If input is not 2D.
        """
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2:
            raise ValueError(f"Input must be 2D array, got shape {x.shape}")

        # Apply 1D-DCT along columns (axis=0), then rows (axis=1)
        # Using orthonormal DCT for energy preservation
        result = dct(dct(x.T, norm="ortho").T, norm="ortho")

        return result

    def idct2d(self, D: ArrayLike) -> np.ndarray:
        """Compute inverse 2D Discrete Cosine Transform using SciPy.

        Parameters
        ----------
        D : array_like
            Input 2D-DCT coefficients. Will be converted to float64.

        Returns
        -------
        np.ndarray
            Reconstructed spatial domain array.

        Raises
        ------
        ValueError
            If input is not 2D.
        """
        D = np.asarray(D, dtype=np.float64)

        if D.ndim != 2:
            raise ValueError(f"Input must be 2D array, got shape {D.shape}")

        # Apply 1D-IDCT along columns (axis=0), then rows (axis=1)
        # Using orthonormal IDCT
        result = idct(idct(D.T, norm="ortho").T, norm="ortho")

        return result

"""Abstract base class for DCT transform backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    import numpy as np

    ArrayLike = Union[np.ndarray, list]


class Backend(ABC):
    """Abstract base class for DCT transform backends.

    Defines the interface for 2D-DCT and inverse DCT operations.
    Implementations can use different backends (NumPy/SciPy, PyTorch, etc.)
    for CPU or GPU acceleration.

    All implementations must use orthonormal DCT (norm='ortho') to ensure:
    - Energy preservation: sum(D^2) = sum(X^2)
    - Exact inverse: idct2d(dct2d(x)) == x
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'numpy', 'torch')."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Device being used (e.g., 'cpu', 'cuda')."""
        ...

    @abstractmethod
    def dct2d(self, x: ArrayLike) -> np.ndarray:
        """Compute 2D Discrete Cosine Transform.

        Parameters
        ----------
        x : array_like
            Input 2D array (image). Will be converted to appropriate dtype.

        Returns
        -------
        np.ndarray
            2D-DCT coefficients with same shape as input.

        Raises
        ------
        ValueError
            If input is not 2D.
        """
        ...

    @abstractmethod
    def idct2d(self, D: ArrayLike) -> np.ndarray:
        """Compute inverse 2D Discrete Cosine Transform.

        Parameters
        ----------
        D : array_like
            Input 2D-DCT coefficients.

        Returns
        -------
        np.ndarray
            Reconstructed spatial domain array.

        Raises
        ------
        ValueError
            If input is not 2D.
        """
        ...

    def is_available(self) -> bool:
        """Check if backend is available.

        Returns
        -------
        bool
            True if backend can be instantiated and used.
        """
        return True

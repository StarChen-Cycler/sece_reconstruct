"""Backend abstraction for DCT transforms.

Provides a unified interface for CPU (NumPy/SciPy) and GPU (PyTorch)
implementations of 2D-DCT transforms.

Example:
    >>> from sece.backends import get_backend
    >>> backend = get_backend('numpy')
    >>> D = backend.dct2d(image)
    >>> recovered = backend.idct2d(D)

    >>> # Use GPU if available
    >>> backend = get_backend('torch')  # Falls back to CPU if no CUDA
    >>> D = backend.dct2d(image)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sece.backends.base import Backend

__all__ = ["get_backend", "Backend"]


def get_backend(name: str = "numpy", *, device: str = "auto") -> Backend:
    """Get a DCT transform backend.

    Parameters
    ----------
    name : str, optional
        Backend name: 'numpy' (CPU) or 'torch' (GPU/CPU), by default 'numpy'
    device : str, optional
        Device for torch backend: 'auto', 'cuda', or 'cpu', by default 'auto'
        When 'auto', uses CUDA if available, otherwise CPU.

    Returns
    -------
    Backend
        Backend instance for DCT transforms.

    Raises
    ------
    ValueError
        If backend name is unknown or import fails.

    Examples
    --------
    >>> backend = get_backend('numpy')
    >>> D = backend.dct2d(image)

    >>> backend = get_backend('torch', device='cuda')
    >>> D = backend.dct2d(image)
    """
    if name == "numpy":
        from sece.backends.numpy_backend import NumpyBackend

        return NumpyBackend()

    if name == "torch":
        try:
            from sece.backends.torch_backend import TorchBackend

            return TorchBackend(device=device)
        except ImportError as e:
            raise ValueError(
                f"PyTorch backend requires torch and torch-dct packages. "
                f"Install with: pip install sece[gpu]. Error: {e}"
            ) from e

    raise ValueError(
        f"Unknown backend: {name}. Supported: 'numpy', 'torch'"
    )

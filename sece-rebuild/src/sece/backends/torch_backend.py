"""PyTorch backend for DCT transforms with GPU support."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from sece.backends.base import Backend

if TYPE_CHECKING:
    from typing import Union

    import torch

    ArrayLike = Union[np.ndarray, list]


class TorchBackend(Backend):
    """PyTorch backend for 2D-DCT transforms with GPU support.

    Uses torch-dct library for GPU-accelerated DCT operations.
    Automatically falls back to CPU if CUDA is unavailable.

    Parameters
    ----------
    device : str, optional
        Device to use: 'auto', 'cuda', or 'cpu', by default 'auto'
        When 'auto', uses CUDA if available, otherwise CPU.

    Attributes
    ----------
    name : str
        Always 'torch'.
    device : str
        Actual device being used ('cuda' or 'cpu').

    Notes
    -----
    Requires torch and torch-dct packages. Install with:
        pip install sece[gpu]

    For 4GB VRAM constraint:
    - 512x512 images use ~5MB VRAM (safe)
    - 1024x1024 images use ~20MB VRAM (safe)
    - 2048x2048 images use ~80MB VRAM (safe)

    Examples
    --------
    >>> backend = TorchBackend()  # Auto-detect CUDA
    >>> D = backend.dct2d(image)
    >>> recovered = backend.idct2d(D)
    >>> np.allclose(image, recovered)
    True

    >>> # Force CPU
    >>> backend = TorchBackend(device='cpu')
    """

    def __init__(self, device: str = "auto") -> None:
        """Initialize PyTorch backend.

        Parameters
        ----------
        device : str, optional
            Device to use: 'auto', 'cuda', or 'cpu', by default 'auto'
        """
        import torch

        self._torch = torch

        # Determine actual device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                warnings.warn(
                    "CUDA not available, falling back to CPU",
                    UserWarning,
                    stacklevel=2,
                )
                self._device = "cpu"
        elif device in ("cuda", "cpu"):
            if device == "cuda" and not torch.cuda.is_available():
                warnings.warn(
                    "CUDA not available, falling back to CPU",
                    UserWarning,
                    stacklevel=2,
                )
                self._device = "cpu"
            else:
                self._device = device
        else:
            raise ValueError(f"Invalid device: {device}. Use 'auto', 'cuda', or 'cpu'")

        # Try to import torch_dct
        try:
            import torch_dct

            self._torch_dct = torch_dct
        except ImportError as e:
            raise ImportError(
                "torch-dct package required for PyTorch backend. "
                "Install with: pip install torch-dct"
            ) from e

    @property
    def name(self) -> str:
        """Backend name."""
        return "torch"

    @property
    def device(self) -> str:
        """Device being used ('cuda' or 'cpu')."""
        return self._device

    def _to_torch(self, x: ArrayLike) -> torch.Tensor:
        """Convert array to PyTorch tensor on device."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Input must be 2D array, got shape {x.shape}")
        return self._torch.from_numpy(x).to(self._device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to NumPy array."""
        return tensor.detach().cpu().numpy()

    def dct2d(self, x: ArrayLike) -> np.ndarray:
        """Compute 2D Discrete Cosine Transform using PyTorch.

        Parameters
        ----------
        x : array_like
            Input 2D array (image). Will be converted to float64 tensor.

        Returns
        -------
        np.ndarray
            2D-DCT coefficients with same shape as input.

        Raises
        ------
        ValueError
            If input is not 2D.
        """
        tensor = self._to_torch(x)

        # torch_dct.dct_2d applies DCT along last two dimensions
        # Using norm='ortho' for energy preservation
        result_tensor = self._torch_dct.dct_2d(tensor, norm="ortho")

        return self._to_numpy(result_tensor)

    def idct2d(self, D: ArrayLike) -> np.ndarray:
        """Compute inverse 2D Discrete Cosine Transform using PyTorch.

        Parameters
        ----------
        D : array_like
            Input 2D-DCT coefficients. Will be converted to float64 tensor.

        Returns
        -------
        np.ndarray
            Reconstructed spatial domain array.

        Raises
        ------
        ValueError
            If input is not 2D.
        """
        tensor = self._to_torch(D)

        # torch_dct.idct_2d applies IDCT along last two dimensions
        # Using norm='ortho' for exact inverse
        result_tensor = self._torch_dct.idct_2d(tensor, norm="ortho")

        return self._to_numpy(result_tensor)

    @classmethod
    def is_available(cls) -> bool:
        """Check if PyTorch backend is available.

        Returns
        -------
        bool
            True if torch and torch-dct are importable.
        """
        try:
            import torch  # noqa: F401
            import torch_dct  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def is_cuda_available(cls) -> bool:
        """Check if CUDA is available for PyTorch.

        Returns
        -------
        bool
            True if CUDA is available.
        """
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @classmethod
    def get_device_info(cls) -> dict[str, str | int]:
        """Get GPU device information.

        Returns
        -------
        dict
            Device info with keys: 'name', 'vram_gb', 'cuda_version'.
            Returns empty dict if CUDA unavailable.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return {}

            props = torch.cuda.get_device_properties(0)
            return {
                "name": props.name,
                "vram_gb": round(props.total_memory / (1024**3), 1),
                "cuda_version": torch.version.cuda or "unknown",
            }
        except ImportError:
            return {}

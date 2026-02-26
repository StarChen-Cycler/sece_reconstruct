"""SECE - Spatial Entropy-based Contrast Enhancement.

This package implements the SECE (Spatial Entropy-based Contrast Enhancement)
and SECEDCT algorithms from Turgay Celik's 2014 IEEE paper for artifact-free
global and local image contrast enhancement.

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast
    Enhancement," IEEE Trans. Image Process., vol. 23, no. 5,
    pp. 2148-2158, May 2014.
    DOI: 10.1109/TIP.2014.2364537
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "SECE-Rebuild Contributors"
__all__ = [
    "__version__",
    # Core functions
    "sece",
    "sece_simple",
    "SECEResult",
    "secedct",
    "secedct_simple",
    "SECEDCTResult",
    # DCT functions
    "dct2d",
    "idct2d",
    # Weighting functions
    "compute_alpha",
    "weight_coefficients",
]


def __getattr__(name: str):
    """Lazy imports to avoid cv2 issues during initial import."""
    if name in ("sece", "sece_simple", "SECEResult"):
        from sece.core import SECEResult, sece, sece_simple
        return locals()[name]
    elif name in ("secedct", "secedct_simple", "SECEDCTResult"):
        from sece.secedct import SECEDCTResult, secedct, secedct_simple
        return locals()[name]
    elif name in ("dct2d", "idct2d"):
        from sece.dct import dct2d, idct2d
        return locals()[name]
    elif name in ("compute_alpha", "weight_coefficients"):
        from sece.weighting import compute_alpha, weight_coefficients_vectorized as weight_coefficients
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

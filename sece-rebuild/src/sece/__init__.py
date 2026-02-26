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
]

# Import main classes when implemented
# from sece.core import SECEEnhancer, SECEDCTEnhancer
# from sece.backends import NumpyBackend, TorchBackend
# from sece.metrics import emeg, gmsd

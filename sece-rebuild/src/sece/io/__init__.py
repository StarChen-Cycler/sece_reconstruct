"""Image I/O utilities for SECE package.

Provides format-aware image loading and saving with automatic format detection,
alpha channel handling, and bit depth conversion.
"""

from __future__ import annotations

from sece.io.reader import load_image, ensure_uint8, SUPPORTED_FORMATS
from sece.io.writer import save_image

__all__ = [
    "load_image",
    "save_image",
    "ensure_uint8",
    "SUPPORTED_FORMATS",
]

"""SECE Baselines Module.

Baseline image enhancement algorithms for comparison with SECE/SECEDCT.

Available Baselines:
    - ghe: Global Histogram Equalization (cv2.equalizeHist wrapper)
    - clahe: Contrast Limited Adaptive Histogram Equalization
    - wthe: Weighted Thresholded Histogram Equalization

Usage:
    >>> from sece.baselines import ghe, clahe, wthe
    >>> import numpy as np
    >>>
    >>> image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    >>>
    >>> # Apply Global Histogram Equalization
    >>> enhanced_ghe = ghe(image)
    >>>
    >>> # Apply CLAHE
    >>> enhanced_clahe = clahe(image)
    >>>
    >>> # Apply Weighted Thresholded HE
    >>> enhanced_wthe = wthe(image)
"""

from sece.baselines.clahe import clahe, clahe_with_params
from sece.baselines.ghe import ghe
from sece.baselines.wthe import wthe, wthe_with_params

__all__ = [
    "ghe",
    "clahe",
    "clahe_with_params",
    "wthe",
    "wthe_with_params",
]

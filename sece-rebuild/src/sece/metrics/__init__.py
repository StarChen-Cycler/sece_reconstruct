"""SECE Metrics Module.

Image quality assessment metrics for contrast enhancement evaluation.

Available Metrics:
    - emeg: Expected Measure of Enhancement by Gradient (contrast measure)
    - ssim: Structural Similarity Index Measure (perceptual similarity)
    - gmsd: Gradient Magnitude Similarity Deviation (distortion measure)

Usage:
    >>> from sece.metrics import emeg, ssim, gmsd
    >>> from sece.metrics import emeg_comparison, ssim_comparison, gmsd_comparison
    >>> import numpy as np
    >>> from sece import sece
    >>>
    >>> # Compute EMEG for a single image
    >>> image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    >>> emeg_value = emeg(image)
    >>>
    >>> # Compare original vs enhanced
    >>> result = sece(image)
    >>> comparison = emeg_comparison(image, result.image)
    >>> print(f"EMEG improvement: {comparison['improvement']:.4f}")
    >>>
    >>> # Compute SSIM
    >>> ssim_value = ssim(image, result.image)
    >>> print(f"SSIM: {ssim_value:.4f}")
    >>>
    >>> # Compute GMSD (distortion measure)
    >>> gmsd_value = gmsd(image, result.image)
    >>> print(f"GMSD: {gmsd_value:.4f}")
"""

from sece.metrics.emeg import emeg, emeg_comparison
from sece.metrics.gmsd import gmsd, gmsd_comparison, gmsd_map
from sece.metrics.ssim import ssim, ssim_comparison, ssim_map

__all__ = [
    "emeg",
    "emeg_comparison",
    "gmsd",
    "gmsd_comparison",
    "gmsd_map",
    "ssim",
    "ssim_comparison",
    "ssim_map",
]

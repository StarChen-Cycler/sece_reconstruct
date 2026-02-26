"""LAB color space processor.

LAB (CIE L*a*b*) is a device-independent color space where:
- L*: Lightness (luminance) - 0 to 100 (scaled to 0-255 in OpenCV)
- a*: Green-Red axis
- b*: Blue-Yellow axis

LAB is useful for scientific applications because:
- Device-independent (based on human perception)
- L* channel is perceptually uniform
- Good for measuring perceptual color differences

OpenCV conversion: COLOR_BGR2LAB / COLOR_LAB2BGR
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from sece.color.processor import ColorProcessor


class LABProcessor(ColorProcessor):
    """LAB (CIE L*a*b*) color space processor.

    Uses the L* (Lightness) channel as luminance for enhancement,
    preserving a* and b* chrominance channels.
    """

    @property
    def name(self) -> str:
        """Return color space name."""
        return "lab"

    def to_luminance(
        self, image: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Extract L channel as luminance from BGR image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input BGR image of shape (H, W, 3).

        Returns
        -------
        Tuple[NDArray[np.uint8], NDArray[np.uint8]]
            (L, ab) where L is luminance (H, W) and ab is
            chrominance (H, W, 2) containing a* and b* channels.
        """
        self.validate_input(image)

        # Convert BGR to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Extract L (luminance) and ab (chrominance)
        luminance = lab[:, :, 0]  # L channel
        chrominance = lab[:, :, 1:]  # a* and b* channels

        return luminance.copy(), chrominance.copy()

    def from_luminance(
        self,
        luminance: NDArray[np.uint8],
        chrominance: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Combine enhanced L with ab chrominance.

        Parameters
        ----------
        luminance : NDArray[np.uint8]
            Enhanced L channel of shape (H, W).
        chrominance : NDArray[np.uint8]
            ab channels of shape (H, W, 2).

        Returns
        -------
        NDArray[np.uint8]
            BGR image of shape (H, W, 3).
        """
        H, W = luminance.shape

        # Reconstruct LAB image
        lab = np.empty((H, W, 3), dtype=np.uint8)
        lab[:, :, 0] = luminance  # L (enhanced)
        lab[:, :, 1:] = chrominance  # a* and b*

        # Convert LAB to BGR
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return bgr

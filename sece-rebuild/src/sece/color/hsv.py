"""HSV color space processor.

HSV (Hue, Saturation, Value) is the default color space for SECE
because the V (Value) channel represents perceptually uniform
luminance, which is ideal for contrast enhancement.

OpenCV conversion: COLOR_BGR2HSV / COLOR_HSV2BGR
- H (Hue): 0-179 (scaled for 8-bit)
- S (Saturation): 0-255
- V (Value/Luminance): 0-255
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from sece.color.processor import ColorProcessor


class HSVProcessor(ColorProcessor):
    """HSV color space processor.

    Uses the V (Value) channel as luminance for enhancement,
    preserving H (Hue) and S (Saturation) channels.

    HSV is the default because:
    - V channel represents perceptually uniform luminance
    - H and S channels encode color information independently
    - Changes to V don't shift colors (hue stays constant)
    """

    @property
    def name(self) -> str:
        """Return color space name."""
        return "hsv"

    def to_luminance(
        self, image: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Extract V channel as luminance from BGR image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input BGR image of shape (H, W, 3).

        Returns
        -------
        Tuple[NDArray[np.uint8], NDArray[np.uint8]]
            (V, HS) where V is luminance (H, W) and HS is
            chrominance (H, W, 2) containing H and S channels.
        """
        self.validate_input(image)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract V (luminance) and HS (chrominance)
        luminance = hsv[:, :, 2]  # V channel
        chrominance = hsv[:, :, :2]  # H and S channels

        return luminance.copy(), chrominance.copy()

    def from_luminance(
        self,
        luminance: NDArray[np.uint8],
        chrominance: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Combine enhanced V with HS chrominance.

        Parameters
        ----------
        luminance : NDArray[np.uint8]
            Enhanced V channel of shape (H, W).
        chrominance : NDArray[np.uint8]
            HS channels of shape (H, W, 2).

        Returns
        -------
        NDArray[np.uint8]
            BGR image of shape (H, W, 3).
        """
        H, W = luminance.shape

        # Reconstruct HSV image
        hsv = np.empty((H, W, 3), dtype=np.uint8)
        hsv[:, :, :2] = chrominance  # H and S
        hsv[:, :, 2] = luminance  # V (enhanced)

        # Convert HSV to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

"""YCbCr color space processor.

YCbCr is a video standard color space where:
- Y: Luma (luminance) component
- Cb: Blue-difference chroma
- Cr: Red-difference chroma

YCbCr is useful for:
- Video processing applications
- JPEG compression (native color space)
- Compatibility with video standards

OpenCV conversion: COLOR_BGR2YCrCb / COLOR_YCrCb2BGR
Note: OpenCV uses YCrCb order (Cr before Cb in array)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from sece.color.processor import ColorProcessor


class YCbCrProcessor(ColorProcessor):
    """YCbCr (YCrCb) color space processor.

    Uses the Y (Luma) channel as luminance for enhancement,
    preserving Cb and Cr chrominance channels.

    Note: OpenCV uses YCrCb order internally, but we store
    chrominance as (Cr, Cb) to match OpenCV's array layout.
    """

    @property
    def name(self) -> str:
        """Return color space name."""
        return "ycbcr"

    def to_luminance(
        self, image: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Extract Y channel as luminance from BGR image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input BGR image of shape (H, W, 3).

        Returns
        -------
        Tuple[NDArray[np.uint8], NDArray[np.uint8]]
            (Y, CrCb) where Y is luminance (H, W) and CrCb is
            chrominance (H, W, 2) containing Cr and Cb channels.
        """
        self.validate_input(image)

        # Convert BGR to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Extract Y (luminance) and CrCb (chrominance)
        luminance = ycrcb[:, :, 0]  # Y channel
        chrominance = ycrcb[:, :, 1:]  # Cr and Cb channels

        return luminance.copy(), chrominance.copy()

    def from_luminance(
        self,
        luminance: NDArray[np.uint8],
        chrominance: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Combine enhanced Y with CrCb chrominance.

        Parameters
        ----------
        luminance : NDArray[np.uint8]
            Enhanced Y channel of shape (H, W).
        chrominance : NDArray[np.uint8]
            CrCb channels of shape (H, W, 2).

        Returns
        -------
        NDArray[np.uint8]
            BGR image of shape (H, W, 3).
        """
        H, W = luminance.shape

        # Reconstruct YCrCb image
        ycrcb = np.empty((H, W, 3), dtype=np.uint8)
        ycrcb[:, :, 0] = luminance  # Y (enhanced)
        ycrcb[:, :, 1:] = chrominance  # Cr and Cb

        # Convert YCrCb to BGR
        bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        return bgr

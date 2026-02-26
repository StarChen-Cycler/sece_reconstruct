"""Abstract base class for color processors.

Defines the interface for color space conversions that support
luminance-based contrast enhancement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class ColorProcessor(ABC):
    """Abstract base class for color space processors.

    Color processors handle conversion between BGR (OpenCV format)
    and a color space suitable for luminance-based enhancement.

    The pattern is:
    1. Convert BGR to target color space
    2. Extract luminance channel
    3. Apply enhancement to luminance
    4. Combine enhanced luminance with chrominance
    5. Convert back to BGR

    Subclasses must implement:
    - to_luminance: Extract luminance and chrominance
    - from_luminance: Combine luminance with chrominance
    - name: Return color space name
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the color space name.

        Returns
        -------
        str
            Color space identifier (e.g., 'hsv', 'lab', 'ycbcr').
        """
        pass

    @abstractmethod
    def to_luminance(
        self, image: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Extract luminance channel from BGR image.

        Parameters
        ----------
        image : NDArray[np.uint8]
            Input BGR image of shape (H, W, 3).

        Returns
        -------
        Tuple[NDArray[np.uint8], NDArray[np.uint8]]
            (luminance, chrominance) tuple where:
            - luminance: Shape (H, W), grayscale for enhancement
            - chrominance: Shape (H, W, 2), color channels to preserve
        """
        pass

    @abstractmethod
    def from_luminance(
        self,
        luminance: NDArray[np.uint8],
        chrominance: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Combine enhanced luminance with chrominance.

        Parameters
        ----------
        luminance : NDArray[np.uint8]
            Enhanced luminance channel of shape (H, W).
        chrominance : NDArray[np.uint8]
            Chrominance channels of shape (H, W, 2).

        Returns
        -------
        NDArray[np.uint8]
            BGR image of shape (H, W, 3).
        """
        pass

    def validate_input(self, image: NDArray) -> None:
        """Validate input image format.

        Parameters
        ----------
        image : NDArray
            Input image to validate.

        Raises
        ------
        ValueError
            If image is not 3-channel uint8 BGR.
        """
        if image.ndim != 3:
            raise ValueError(
                f"Color image must be 3D (H, W, 3), got shape {image.shape}"
            )
        if image.shape[2] != 3:
            raise ValueError(
                f"Color image must have 3 channels, got {image.shape[2]}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"Color image must be uint8, got {image.dtype}"
            )

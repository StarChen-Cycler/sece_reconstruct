# SECE API Reference

Complete API documentation for the SECE (Spatial Entropy-based Contrast Enhancement) library.

## Table of Contents

- [Core Functions](#core-functions)
  - [sece()](#sece)
  - [sece_simple()](#sece_simple)
  - [secedct()](#secedct)
  - [secedct_simple()](#secedct_simple)
- [Color Functions](#color-functions)
  - [color_sece()](#color_sece)
  - [color_secedct()](#color_secedct)
- [Result Classes](#result-classes)
  - [SECEResult](#seceresult)
  - [SECEDCTResult](#secedctresult)
  - [ColorSECEResult](#colorseceresult)
  - [ColorSECEDCTResult](#colorsecedctresult)
- [DCT Functions](#dct-functions)
  - [dct2d()](#dct2d)
  - [idct2d()](#idct2d)
- [Weighting Functions](#weighting-functions)
  - [compute_alpha()](#compute_alpha)
  - [weight_coefficients()](#weight_coefficients)
- [Metrics](#metrics)
  - [emeg()](#emeg)
  - [ssim()](#ssim)
  - [gmsd()](#gmsd)
- [Baselines](#baselines)
  - [ghe()](#ghe)
  - [clahe()](#clahe)
  - [wthe()](#wthe)
- [Backends](#backends)
  - [get_backend()](#get_backend)
- [I/O Functions](#io-functions)
  - [load_image()](#load_image)
  - [save_image()](#save_image)

---

## Core Functions

### sece()

```python
def sece(
    image: NDArray[np.uint8],
    y_d: int = 0,
    y_u: int = 255,
    epsilon: float = 1e-10,
) -> SECEResult
```

Apply SECE (Spatial Entropy-based Contrast Enhancement) for global contrast enhancement.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | `NDArray[np.uint8]` | required | Input grayscale image of shape (H, W) |
| `y_d` | `int` | `0` | Lower bound of output range |
| `y_u` | `int` | `255` | Upper bound of output range |
| `epsilon` | `float` | `1e-10` | Small value for numerical stability |

**Returns:** `SECEResult` - Contains enhanced image, distribution, and metadata.

**Raises:**
- `ValueError` - If image is not 2D grayscale uint8.

**Example:**

```python
import cv2
from sece import sece

image = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
result = sece(image)
enhanced = result.image
```

---

### sece_simple()

```python
def sece_simple(
    image: NDArray[np.uint8],
    y_d: int = 0,
    y_u: int = 255,
) -> NDArray[np.uint8]
```

Simplified SECE that returns only the enhanced image.

**Parameters:** Same as `sece()` except `epsilon`.

**Returns:** `NDArray[np.uint8]` - Enhanced image.

**Example:**

```python
from sece import sece_simple

enhanced = sece_simple(image)
```

---

### secedct()

```python
def secedct(
    image: NDArray[np.uint8],
    gamma: float = 0.5,
    y_d: int = 0,
    y_u: int = 255,
    epsilon: float = 1e-10,
) -> SECEDCTResult
```

Apply SECEDCT for combined global and local contrast enhancement using DCT coefficient weighting.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | `NDArray[np.uint8]` | required | Input grayscale image of shape (H, W) |
| `gamma` | `float` | `0.5` | Local enhancement level [0, 1] |
| `y_d` | `int` | `0` | Lower bound of output range |
| `y_u` | `int` | `255` | Upper bound of output range |
| `epsilon` | `float` | `1e-10` | Small value for numerical stability |

**Returns:** `SECEDCTResult` - Contains enhanced image, SECE result, alpha, gamma.

**Raises:**
- `ValueError` - If image is not 2D grayscale uint8 or gamma out of range.

**Gamma Values:**
- `0.0`: No local enhancement (output equals SECE)
- `0.5`: Default balanced enhancement
- `1.0`: Maximum local enhancement

**Example:**

```python
from sece import secedct

# Default gamma (0.5)
result = secedct(image)

# Maximum local enhancement
result = secedct(image, gamma=1.0)
```

---

### secedct_simple()

```python
def secedct_simple(
    image: NDArray[np.uint8],
    gamma: float = 0.5,
    y_d: int = 0,
    y_u: int = 255,
) -> NDArray[np.uint8]
```

Simplified SECEDCT that returns only the enhanced image.

**Example:**

```python
from sece import secedct_simple

enhanced = secedct_simple(image, gamma=0.7)
```

---

## Color Functions

### color_sece()

```python
def color_sece(
    image: NDArray[np.uint8],
    color_space: ColorSpace = "hsv",
    y_d: int = 0,
    y_u: int = 255,
) -> ColorSECEResult
```

Apply SECE to color images. Converts to specified color space, enhances luminance channel, and converts back to BGR.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | `NDArray[np.uint8]` | required | Input BGR image of shape (H, W, 3) |
| `color_space` | `ColorSpace` | `"hsv"` | Color space: `"hsv"`, `"lab"`, or `"ycbcr"` |
| `y_d` | `int` | `0` | Lower bound of output range |
| `y_u` | `int` | `255` | Upper bound of output range |

**Returns:** `ColorSECEResult` - Contains enhanced BGR image and intermediate results.

**Color Spaces:**
- `"hsv"`: Perceptually uniform, preserves hue (default)
- `"lab"`: Device-independent, good for scientific applications
- `"ycbcr"`: Video standard, good for JPEG-like processing

**Example:**

```python
import cv2
from sece import color_sece

image = cv2.imread("photo.jpg")  # BGR format

# Default HSV
result = color_sece(image)

# LAB color space
result = color_sece(image, color_space="lab")

enhanced = result.image
```

---

### color_secedct()

```python
def color_secedct(
    image: NDArray[np.uint8],
    gamma: float = 0.5,
    color_space: ColorSpace = "hsv",
    y_d: int = 0,
    y_u: int = 255,
) -> ColorSECEDCTResult
```

Apply SECEDCT to color images with combined global and local enhancement.

**Example:**

```python
from sece import color_secedct

result = color_secedct(image, gamma=0.7, color_space="hsv")
enhanced = result.image
```

---

## Result Classes

### SECEResult

```python
@dataclass
class SECEResult:
    image: NDArray[np.uint8]      # Enhanced image (H, W)
    distribution: NDArray[np.float64]  # Distribution f_k for each gray level
    gray_levels: NDArray[np.int64]     # Input gray levels
    cdf: NDArray[np.float64]          # Cumulative distribution function F
    processing_time_ms: float          # Processing time in milliseconds
```

Result from `sece()` function.

---

### SECEDCTResult

```python
@dataclass
class SECEDCTResult:
    image: NDArray[np.uint8]      # Enhanced image (H, W)
    sece_result: SECEResult       # Intermediate SECE result
    alpha: float                  # DCT weighting parameter
    gamma: float                  # Local enhancement level [0, 1]
    processing_time_ms: float     # Processing time in milliseconds
```

Result from `secedct()` function.

---

### ColorSECEResult

```python
@dataclass
class ColorSECEResult:
    image: NDArray[np.uint8]      # Enhanced BGR image (H, W, 3)
    sece_result: SECEResult       # Intermediate SECE result from luminance
    color_space: str              # Color space used for processing
    processing_time_ms: float     # Processing time in milliseconds
```

Result from `color_sece()` function.

---

### ColorSECEDCTResult

```python
@dataclass
class ColorSECEDCTResult:
    image: NDArray[np.uint8]         # Enhanced BGR image (H, W, 3)
    secedct_result: SECEDCTResult    # Intermediate SECEDCT result from luminance
    color_space: str                 # Color space used for processing
    processing_time_ms: float        # Processing time in milliseconds
```

Result from `color_secedct()` function.

---

## DCT Functions

### dct2d()

```python
def dct2d(
    x: ArrayLike,
    norm: str = "ortho",
) -> NDArray[np.float64]
```

Compute 2D Discrete Cosine Transform (Type II).

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `x` | `ArrayLike` | required | Input 2D array |
| `norm` | `str` | `"ortho"` | Normalization mode |

**Returns:** `NDArray[np.float64]` - DCT coefficients.

**Example:**

```python
from sece import dct2d, idct2d

D = dct2d(image.astype(np.float64))
recovered = idct2d(D)
```

---

### idct2d()

```python
def idct2d(
    D: ArrayLike,
    norm: str = "ortho",
) -> NDArray[np.float64]
```

Compute inverse 2D DCT (Type III).

---

## Weighting Functions

### compute_alpha()

```python
def compute_alpha(
    distribution: ArrayLike,
    gamma: float = 0.5,
    epsilon: float = 1e-10,
) -> float
```

Compute alpha parameter for DCT coefficient weighting using formula: α = entropy(f)^γ

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `distribution` | `ArrayLike` | required | Distribution function f |
| `gamma` | `float` | `0.5` | Exponent for entropy |
| `epsilon` | `float` | `1e-10` | Numerical stability constant |

**Returns:** `float` - Alpha value in [1, K] where K is number of levels.

---

### weight_coefficients()

```python
def weight_coefficients(
    D: ArrayLike,
    alpha: float,
) -> NDArray[np.float64]
```

Weight DCT coefficients using formula: w(k,l) = 1 + (α-1) * (k+l)/(K+L-2)

Higher frequencies get higher weights, enhancing local contrast.

---

## Metrics

### emeg()

```python
def emeg(
    image: ImageArray,
    block_size: int = 8,
    epsilon: float = 1e-10,
) -> float
```

Compute EMEG (Enhancement Measure by Entropy of Gradient).

Higher values indicate better contrast enhancement.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | `ImageArray` | required | Input image (grayscale or color) |
| `block_size` | `int` | `8` | Block size for local computation |
| `epsilon` | `float` | `1e-10` | Numerical stability constant |

**Returns:** `float` - EMEG value (typically 0 to 1).

---

### ssim()

```python
def ssim(
    reference: ImageArray,
    distorted: ImageArray,
    win_size: int | None = None,
    data_range: float | None = None,
) -> float
```

Compute SSIM (Structural Similarity Index) between two images.

**Returns:** `float` - SSIM value (1.0 = identical, 0 = completely different).

---

### gmsd()

```python
def gmsd(
    reference: ImageArray,
    distorted: ImageArray,
) -> float
```

Compute GMSD (Gradient Magnitude Similarity Deviation).

**Returns:** `float` - GMSD value (0 = identical, higher = more distortion).

---

## Baselines

### ghe()

```python
def ghe(
    image: ImageArray,
) -> ImageArray
```

Global Histogram Equalization using OpenCV.

---

### clahe()

```python
def clahe(
    image: ImageArray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> ImageArray
```

Contrast Limited Adaptive Histogram Equalization.

---

### wthe()

```python
def wthe(
    image: ImageArray,
    r: float = 0.5,
    v: float = 0.5,
) -> ImageArray
```

Weighted Thresholded Histogram Equalization.

---

## Backends

### get_backend()

```python
def get_backend(
    name: str = "numpy",
    device: str = "cpu",
) -> Backend
```

Get DCT backend for computations.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | `"numpy"` | Backend name: `"numpy"` or `"torch"` |
| `device` | `str` | `"cpu"` | Device: `"cpu"` or `"cuda"` |

**Returns:** `Backend` - Backend instance with `dct2d()` and `idct2d()` methods.

**Example:**

```python
from sece.backends import get_backend

# NumPy CPU backend (default)
backend = get_backend("numpy")

# PyTorch GPU backend
backend = get_backend("torch", device="cuda")

D = backend.dct2d(image.astype(np.float64))
```

---

## I/O Functions

### load_image()

```python
def load_image(
    path: str | Path,
    color_mode: str = "auto",
) -> ImageArray
```

Load image with automatic format detection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str \| Path` | required | Path to image file |
| `color_mode` | `str` | `"auto"` | `"auto"`, `"grayscale"`, or `"color"` |

**Supported Formats:** PNG, JPEG, TIFF, BMP, WebP

---

### save_image()

```python
def save_image(
    image: ImageArray,
    path: str | Path,
    quality: int = 95,
) -> bool
```

Save image to file with format detection from extension.

---

## Exceptions

```python
class SECEError(Exception):
    """Base exception for SECE library."""

class ImageLoadError(SECEError):
    """Failed to load image."""

class ImageSaveError(SECEError):
    """Failed to save image."""

class UnsupportedFormatError(SECEError):
    """Unsupported image format."""

class UnsupportedColorSpaceError(SECEError):
    """Unsupported color space."""

class InvalidParameterError(SECEError):
    """Invalid parameter value."""
```

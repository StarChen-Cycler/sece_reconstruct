# SECE - Spatial Entropy-based Contrast Enhancement

[![PyPI version](https://badge.fury.io/py/sece.svg)](https://badge.fury.io/py/sece)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of SECE (Spatial Entropy-based Contrast Enhancement) and SECEDCT algorithms from Turgay Celik's 2014 IEEE paper for artifact-free global and local image contrast enhancement.

## Key Features

- **Artifact-free enhancement**: Preserves histogram shape unlike traditional histogram equalization
- **Global + Local contrast**: SECE for global, SECEDCT for combined global+local enhancement
- **Color support**: HSV, LAB, YCbCr color spaces with hue preservation
- **Multiple backends**: NumPy (CPU) and PyTorch (GPU optional) support
- **Quality metrics**: Built-in EMEG, SSIM, GMSD for evaluation
- **CLI tool**: Command-line interface for batch processing

## Installation

```bash
pip install sece
```

For development with GPU support:

```bash
pip install -e ".[dev,gpu]"
```

### Dependencies

- Python >= 3.10
- NumPy >= 1.20
- SciPy >= 1.7
- OpenCV >= 4.5
- scikit-image >= 0.19

Optional:
- PyTorch >= 2.0 (GPU acceleration)
- Click >= 8.0 (CLI)
- Rich >= 13.0 (CLI output)

## Quick Start

### Grayscale Enhancement

```python
import cv2
from sece import sece, secedct, sece_simple, secedct_simple

# Load grayscale image
image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Global contrast enhancement (SECE)
enhanced = sece_simple(image)

# Global + Local contrast enhancement (SECEDCT)
enhanced_local = secedct_simple(image, gamma=0.5)
```

### Color Enhancement

```python
import cv2
from sece import color_sece, color_secedct

# Load color image (BGR format)
image = cv2.imread("photo.jpg")

# HSV-based enhancement (default, perceptually uniform)
result = color_sece(image, color_space="hsv")

# LAB-based enhancement (device-independent)
result = color_sece(image, color_space="lab")

# Get enhanced image
enhanced = result.image
```

### Detailed Results

```python
from sece import sece, secedct

# Get full results including distribution data
result = sece(image)

print(f"Processing time: {result.processing_time_ms:.2f}ms")
print(f"Gray levels: {len(result.gray_levels)}")
print(f"CDF range: [{result.cdf.min():.3f}, {result.cdf.max():.3f}]")

enhanced = result.image
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `sece(image, y_d=0, y_u=255)` | Global contrast enhancement, returns `SECEResult` |
| `sece_simple(image)` | Simplified SECE, returns only enhanced image |
| `secedct(image, gamma=0.5, y_d=0, y_u=255)` | Global+local enhancement, returns `SECEDCTResult` |
| `secedct_simple(image, gamma=0.5)` | Simplified SECEDCT, returns only enhanced image |

### Color Functions

| Function | Description |
|----------|-------------|
| `color_sece(image, color_space="hsv")` | Color SECE, returns `ColorSECEResult` |
| `color_secedct(image, gamma=0.5, color_space="hsv")` | Color SECEDCT, returns `ColorSECEDCTResult` |

### Result Classes

```python
@dataclass
class SECEResult:
    image: np.ndarray          # Enhanced image (H, W) uint8
    distribution: np.ndarray   # Distribution f_k for each gray level
    gray_levels: np.ndarray    # Input gray levels
    cdf: np.ndarray           # Cumulative distribution function F
    processing_time_ms: float  # Processing time in milliseconds

@dataclass
class SECEDCTResult:
    image: np.ndarray          # Enhanced image (H, W) uint8
    sece_result: SECEResult    # Intermediate SECE result
    alpha: float              # DCT weighting parameter
    gamma: float              # Local enhancement level [0, 1]
    processing_time_ms: float  # Processing time in milliseconds
```

### Color Spaces

| Space | Description | Use Case |
|-------|-------------|----------|
| `hsv` | HSV color space (default) | Perceptually uniform, general purpose |
| `lab` | CIE LAB color space | Device-independent, scientific applications |
| `ycbcr` | YCbCr color space | Video standard, JPEG-like processing |

## Parameters

### gamma (SECEDCT)

Controls local contrast enhancement level:
- `gamma=0`: No local enhancement, output equals SECE result
- `gamma=0.5`: Default, balanced global+local enhancement
- `gamma=1`: Maximum local contrast enhancement

### Output Range (y_d, y_u)

Specify output dynamic range:
- Default: `[0, 255]` (full 8-bit range)
- Custom: `sece(image, y_d=10, y_u=245)` to avoid extreme values

## CLI Usage

```bash
# Single image enhancement
sece enhance input.png output.png

# SECEDCT with gamma
sece enhance input.png output.png --algorithm secedct --gamma 0.7

# Color image enhancement
sece enhance photo.jpg enhanced.jpg --color-space hsv

# Batch processing
sece enhance ./images ./output --batch

# With quality metrics
sece enhance input.png output.png --metrics emeg,ssim,gmsd

# Verbose output
sece enhance input.png output.png --verbose
```

## Quality Metrics

```python
from sece.metrics import emeg, ssim, gmsd

# EMEG: Enhancement measure (higher = better contrast)
score = emeg(enhanced_image)

# SSIM: Structural similarity (1.0 = identical)
similarity = ssim(original, enhanced)

# GMSD: Gradient magnitude deviation (0 = identical)
distortion = gmsd(original, enhanced)

# Comparison helper
from sece.metrics import emeg_comparison, gmsd_comparison
comparison = emeg_comparison(original, enhanced)
print(f"Improvement: {comparison['improvement']:.2%}")
```

## Baseline Algorithms

```python
from sece.baselines import ghe, clahe, wthe

# Global Histogram Equalization
enhanced = ghe(image)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
enhanced = clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))

# Weighted Thresholded Histogram Equalization
enhanced = wthe(image, r=0.5, v=0.5)
```

## DCT Backend Selection

```python
from sece.backends import get_backend

# NumPy backend (CPU, default)
backend = get_backend("numpy")

# PyTorch backend (GPU if available)
backend = get_backend("torch", device="cuda")

# Use backend for custom DCT operations
D = backend.dct2d(image.astype(np.float64))
recovered = backend.idct2d(D)
```

## Algorithm Overview

### SECE (Global Enhancement)

1. Compute spatial histograms for all gray levels (Formula 1)
2. Compute grid size M×N preserving aspect ratio (Formula 2)
3. Compute spatial entropy S_k for each level (Formula 3)
4. Compute distribution function f and CDF F (Formulas 4-6)
5. Map gray levels using CDF (Formula 7)

### SECEDCT (Global + Local Enhancement)

1. Apply SECE to get globally enhanced image Y_global
2. Compute DCT of Y_global: D = DCT(Y_global)
3. Calculate alpha = entropy(f)^gamma
4. Weight DCT coefficients using alpha (higher frequencies weighted more)
5. Apply inverse DCT: Y_final = IDCT(D_weighted)

## Performance

Typical processing times on AMD Ryzen 7 5800H:

| Image Size | SECE | SECEDCT |
|------------|------|---------|
| 256×256 | ~15ms | ~25ms |
| 512×512 | ~50ms | ~80ms |
| 1024×1024 | ~180ms | ~300ms |

GPU acceleration (RTX 3050 Ti) provides ~2-3x speedup for large images.

## Reference

T. Celik, "Spatial Entropy-Based Global and Local Image Contrast Enhancement," IEEE Trans. Image Process., vol. 23, no. 5, pp. 2148-2158, May 2014. DOI: [10.1109/TIP.2014.2364537](https://doi.org/10.1109/TIP.2014.2364537)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read the development guidelines in `CLAUDE.md` and ensure all tests pass:

```bash
pytest tests/ -v --cov=sece
```

## Changelog

### v0.1.0 (2026-02)

- Initial release
- SECE global contrast enhancement
- SECEDCT global+local contrast enhancement
- Color support (HSV, LAB, YCbCr)
- Quality metrics (EMEG, SSIM, GMSD)
- Baseline algorithms (GHE, CLAHE, WTHE)
- CLI tool with batch processing
- NumPy and PyTorch backends

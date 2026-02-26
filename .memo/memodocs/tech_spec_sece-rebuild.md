# Technical Specification: SECE-Rebuild

**Project**: sece-rebuild
**Version**: 0.1.0
**Python**: >=3.10
**Last Updated**: 2026-02-27

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│  (click + rich for command-line interface)                       │
├─────────────────────────────────────────────────────────────────┤
│                       API Layer                                  │
│  SECEEnhancer, SECEDCTEnhancer, EnhancerFactory                  │
├─────────────────────────────────────────────────────────────────┤
│                    Processing Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   SECE      │  │  SECEDCT    │  │   Color     │              │
│  │   Core      │  │  Transform  │  │  Processing │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                    Backend Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │  NumPy Backend  │  │  Torch Backend  │                       │
│  │    (CPU)        │  │    (GPU)        │                       │
│  └─────────────────┘  └─────────────────┘                       │
├─────────────────────────────────────────────────────────────────┤
│                   Utilities Layer                                │
│  I/O, Metrics, Baselines, Validation                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20 | Array operations, numerical computing |
| scipy | >=1.7 | DCT/IDCT transforms (fftpack) |
| opencv-python | >=4.5 | Image I/O, color space conversion |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0 | GPU acceleration backend |
| click | >=8.0 | CLI framework |
| rich | >=13.0 | Progress bars, formatted output |
| piq | >=0.7 | Production IQA metrics (optional validation) |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=7.0 | Testing framework |
| pytest-cov | >=4.0 | Coverage reporting |
| black | >=23.0 | Code formatting |
| mypy | >=1.0 | Type checking |
| ruff | >=0.1.0 | Linting |
| sphinx | >=7.0 | Documentation |
| furo | >=2024.0 | Sphinx theme |

---

## Module Design

### 1. Core Module (`sece/core/`)

#### 1.1 SECE Algorithm (`sece.py`)

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import numpy as np

class BaseEnhancer(ABC):
    """Abstract base class for all enhancers."""

    def __init__(
        self,
        y_d: int = 0,
        y_u: int = 255,
        backend: str = 'numpy'
    ):
        self.y_d = y_d
        self.y_u = y_u
        self._backend = self._get_backend(backend)

    @abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance a grayscale image."""
        pass

    def _get_backend(self, name: str):
        if name == 'numpy':
            from sece.backends.numpy_backend import NumpyBackend
            return NumpyBackend()
        elif name == 'torch':
            from sece.backends.torch_backend import TorchBackend
            return TorchBackend()
        raise ValueError(f"Unknown backend: {name}")


class SECEEnhancer(BaseEnhancer):
    """Spatial Entropy-based Contrast Enhancement."""

    def __init__(
        self,
        y_d: int = 0,
        y_u: int = 255,
        backend: str = 'numpy',
        color_space: str = 'hsv'
    ):
        super().__init__(y_d, y_u, backend)
        self.color_space = color_space
        self._distribution_f: Optional[np.ndarray] = None

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply SECE enhancement to image.

        Parameters
        ----------
        image : np.ndarray
            Input grayscale image (H, W) or color image (H, W, 3)

        Returns
        -------
        np.ndarray
            Enhanced image with same shape as input
        """
        if image.ndim == 3:
            return self._enhance_color(image)

        # Core SECE algorithm
        x_levels, K = self._get_gray_levels(image)
        M, N = self._compute_grid_size(K, image.shape)

        # Spatial entropy for each gray level
        S = np.zeros(K)
        for idx, x_k in enumerate(x_levels):
            h_k = self._compute_spatial_histogram(image, x_k, M, N)
            S[idx] = self._compute_spatial_entropy(h_k)

        # Distribution function
        f, F = self._compute_distribution(S)
        self._distribution_f = f

        # Mapping
        y_levels = self._mapping_function(F)
        Y = self._apply_mapping(image, x_levels, y_levels)

        return Y

    def _get_gray_levels(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Get sorted unique gray levels and count."""
        levels = np.unique(image)
        return np.sort(levels), len(levels)

    def _compute_grid_size(self, K: int, shape: Tuple[int, int]) -> Tuple[int, int]:
        """Formula (2): Compute M×N grid size preserving aspect ratio."""
        H, W = shape[:2]
        r = H / W
        N = int(np.floor(np.sqrt(K / r)))
        M = int(np.floor(np.sqrt(K * r)))
        # Ensure minimum grid size
        M = max(M, 1)
        N = max(N, 1)
        return M, N

    def _compute_spatial_histogram(
        self,
        image: np.ndarray,
        x_k: float,
        M: int,
        N: int
    ) -> np.ndarray:
        """Formula (1): 2D spatial histogram for gray level x_k."""
        H, W = image.shape
        h_k = np.zeros((M, N))

        cell_h = H / M
        cell_w = W / N

        for m in range(M):
            for n in range(N):
                r_start = int(m * cell_h)
                r_end = int((m + 1) * cell_h)
                c_start = int(n * cell_w)
                c_end = int((n + 1) * cell_w)

                region = image[r_start:r_end, c_start:c_end]
                h_k[m, n] = np.sum(region == x_k)

        return h_k

    def _compute_spatial_entropy(self, h_k: np.ndarray) -> float:
        """Formula (3): Shannon entropy from spatial histogram."""
        total = np.sum(h_k)
        if total == 0:
            return 0.0

        p = h_k / total
        p = p[p > 0]  # Only non-zero probabilities

        return -np.sum(p * np.log2(p))

    def _compute_distribution(
        self,
        S: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Formulas (4)-(6): Compute distribution function f and CDF F."""
        K = len(S)
        f = np.zeros(K)

        total_S = np.sum(S)
        for k in range(K):
            if total_S - S[k] > 0:
                f[k] = S[k] / (total_S - S[k])
            else:
                f[k] = 0.0

        # Normalize (Formula 5)
        f = f / np.sum(f)

        # CDF (Formula 6)
        F = np.cumsum(f)

        return f, F

    def _mapping_function(self, F: np.ndarray) -> np.ndarray:
        """Formula (7): Map CDF to output gray levels."""
        y = np.floor(F * (self.y_u - self.y_d) + self.y_d)
        return np.clip(y, self.y_d, self.y_u).astype(np.uint8)

    def _apply_mapping(
        self,
        image: np.ndarray,
        x_levels: np.ndarray,
        y_levels: np.ndarray
    ) -> np.ndarray:
        """Apply gray level mapping to image."""
        Y = np.zeros_like(image, dtype=np.uint8)
        for idx, x_k in enumerate(x_levels):
            Y[image == x_k] = y_levels[idx]
        return Y

    def get_distribution(self) -> Optional[np.ndarray]:
        """Return the distribution function f from last enhancement."""
        return self._distribution_f
```

#### 1.2 SECEDCT Algorithm (`secedct.py`)

```python
from typing import Optional
import numpy as np
from sece.core.sece import SECEEnhancer

class SECEDCTEnhancer(SECEEnhancer):
    """Spatial Entropy-based Contrast Enhancement in DCT domain."""

    def __init__(
        self,
        gamma: float = 0.5,
        y_d: int = 0,
        y_u: int = 255,
        backend: str = 'numpy',
        color_space: str = 'hsv'
    ):
        super().__init__(y_d, y_u, backend, color_space)
        self.gamma = gamma

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply SECEDCT enhancement (global + local).

        gamma=0: equivalent to SECE
        gamma=1: maximum local enhancement
        """
        # Step 1: Global enhancement with SECE
        Y_global = super().enhance(image)

        # If gamma=0, return SECE result
        if self.gamma == 0:
            return Y_global

        # Step 2: Compute alpha from distribution
        f = self.get_distribution()
        alpha = self._compute_alpha(f)

        # Step 3: 2D-DCT transform
        D = self._backend.dct2d(Y_global.astype(np.float64))

        # Step 4: Weight coefficients
        D_weighted = self._weight_coefficients(D, alpha)

        # Step 5: Inverse 2D-DCT
        Y = self._backend.idct2d(D_weighted)

        # Step 6: Post-processing
        Y = np.clip(Y, self.y_d, self.y_u).astype(np.uint8)

        return Y

    def _compute_alpha(self, f: np.ndarray) -> float:
        """Formula (12): Compute alpha from distribution entropy."""
        f_pos = f[f > 0]
        entropy = -np.sum(f_pos * np.log2(f_pos))
        return entropy ** self.gamma

    def _weight_coefficients(
        self,
        D: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """Formulas (10)-(11): Weight DCT coefficients."""
        H, W = D.shape
        D_weighted = np.zeros_like(D)

        for k in range(H):
            for l in range(W):
                # Formula (11): w(k,l)
                w_kl = (1 + (alpha - 1) * k / (H - 1)) * \
                       (1 + (alpha - 1) * l / (W - 1))
                D_weighted[k, l] = w_kl * D[k, l]

        return D_weighted
```

### 2. Backend Module (`sece/backends/`)

```python
# backends/base.py
from abc import ABC, abstractmethod
from typing import Union
import numpy as np

class Backend(ABC):
    """Abstract backend for numerical operations."""

    @abstractmethod
    def dct2d(self, x: np.ndarray) -> np.ndarray:
        """Forward 2D-DCT transform."""
        pass

    @abstractmethod
    def idct2d(self, x: np.ndarray) -> np.ndarray:
        """Inverse 2D-DCT transform."""
        pass


# backends/numpy_backend.py
from scipy.fftpack import dct, idct
import numpy as np
from sece.backends.base import Backend

class NumpyBackend(Backend):
    """NumPy/SciPy-based CPU backend."""

    def dct2d(self, x: np.ndarray) -> np.ndarray:
        """2D-DCT using scipy.fftpack."""
        return dct(dct(x.T, norm='ortho').T, norm='ortho')

    def idct2d(self, x: np.ndarray) -> np.ndarray:
        """Inverse 2D-DCT using scipy.fftpack."""
        return idct(idct(x.T, norm='ortho').T, norm='ortho')


# backends/torch_backend.py
import numpy as np
from sece.backends.base import Backend

class TorchBackend(Backend):
    """PyTorch-based GPU backend."""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        import torch
        self.torch = torch

    def dct2d(self, x: np.ndarray) -> np.ndarray:
        """2D-DCT on GPU."""
        import torch.nn.functional as F
        x_tensor = self.torch.from_numpy(x).float().to(self.device)
        # Custom DCT implementation for PyTorch
        result = self._dct2d_torch(x_tensor)
        return result.cpu().numpy()

    def idct2d(self, x: np.ndarray) -> np.ndarray:
        """Inverse 2D-DCT on GPU."""
        x_tensor = self.torch.from_numpy(x).float().to(self.device)
        result = self._idct2d_torch(x_tensor)
        return result.cpu().numpy()

    def _dct2d_torch(self, x):
        """PyTorch implementation of 2D-DCT."""
        # Implementation using DCT matrix multiplication
        ...

    def _idct2d_torch(self, x):
        """PyTorch implementation of inverse 2D-DCT."""
        ...
```

### 3. Color Module (`sece/color/`)

```python
# color/processor.py
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import cv2

class ColorProcessor(ABC):
    """Abstract color space processor."""

    @abstractmethod
    def to_luminance(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract luminance channel, return (luminance, chrominance)."""
        pass

    @abstractmethod
    def from_luminance(
        self,
        luminance: np.ndarray,
        chrominance: np.ndarray
    ) -> np.ndarray:
        """Combine luminance with chrominance back to color image."""
        pass


class HSVProcessor(ColorProcessor):
    """HSV color space processor."""

    def to_luminance(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return v, np.stack([h, s], axis=-1)

    def from_luminance(
        self,
        luminance: np.ndarray,
        chrominance: np.ndarray
    ) -> np.ndarray:
        h, s = chrominance[..., 0], chrominance[..., 1]
        hsv = cv2.merge([h, s, luminance])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class LABProcessor(ColorProcessor):
    """LAB color space processor."""

    def to_luminance(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return l, np.stack([a, b], axis=-1)

    def from_luminance(
        self,
        luminance: np.ndarray,
        chrominance: np.ndarray
    ) -> np.ndarray:
        a, b = chrominance[..., 0], chrominance[..., 1]
        lab = cv2.merge([luminance, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class YCbCrProcessor(ColorProcessor):
    """YCbCr color space processor."""

    def to_luminance(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        return y, np.stack([cr, cb], axis=-1)

    def from_luminance(
        self,
        luminance: np.ndarray,
        chrominance: np.ndarray
    ) -> np.ndarray:
        cr, cb = chrominance[..., 0], chrominance[..., 1]
        ycrcb = cv2.merge([luminance, cr, cb])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
```

### 4. CLI Module (`sece/cli/`)

```python
# cli/main.py
import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from pathlib import Path
from typing import Optional
import cv2

console = Console()

@click.group()
def cli():
    """SECE - Spatial Entropy-based Contrast Enhancement."""
    pass


@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), required=True,
              help='Output file or directory path')
@click.option('--method', type=click.Choice(['sece', 'secedct']),
              default='secedct', help='Enhancement method')
@click.option('--gamma', type=float, default=0.5,
              help='Gamma parameter for SECEDCT (0-1)')
@click.option('--color-space', type=click.Choice(['hsv', 'lab', 'ycbcr']),
              default='hsv', help='Color space for processing')
@click.option('--format', 'fmt', type=str, default=None,
              help='Output format (png, jpg, tiff)')
@click.option('--device', type=click.Choice(['cpu', 'cuda']),
              default='cpu', help='Processing device')
@click.option('--metrics', type=str, default=None,
              help='Comma-separated metrics to compute (emeg,gmsd,ssim)')
@click.option('--compare', type=str, default=None,
              help='Comma-separated baselines to compare (ghe,wthe,clahe)')
def enhance(
    input: str,
    output: str,
    method: str,
    gamma: float,
    color_space: str,
    fmt: Optional[str],
    device: str,
    metrics: Optional[str],
    compare: Optional[str]
):
    """Enhance image contrast using SECE or SECEDCT."""
    input_path = Path(input)
    output_path = Path(output)

    # Single image or batch
    if input_path.is_file():
        _process_single(input_path, output_path, method, gamma,
                       color_space, fmt, device, metrics, compare)
    else:
        _process_batch(input_path, output_path, method, gamma,
                      color_space, fmt, device, metrics, compare)


def _process_single(input_path, output_path, method, gamma,
                   color_space, fmt, device, metrics, compare):
    """Process a single image."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Processing {input_path.name}...", total=None)

        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            raise click.ClickException(f"Failed to load image: {input_path}")

        # Create enhancer
        from sece import SECEEnhancer, SECEDCTEnhancer
        if method == 'sece':
            enhancer = SECEEnhancer(backend=device, color_space=color_space)
        else:
            enhancer = SECEDCTEnhancer(gamma=gamma, backend=device,
                                       color_space=color_space)

        # Enhance
        result = enhancer.enhance(image)

        # Compute metrics if requested
        if metrics:
            _compute_and_display_metrics(image, result, metrics)

        # Compare with baselines if requested
        if compare:
            _compare_with_baselines(image, result, compare, gamma)

        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

        progress.update(task, description=f"Saved to {output_path}")


def _process_batch(input_dir, output_dir, method, gamma,
                  color_space, fmt, device, metrics, compare):
    """Process a batch of images."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    images = [f for f in input_dir.rglob('*')
              if f.suffix.lower() in image_extensions]

    with Progress(console=console) as progress:
        task = progress.add_task("Processing batch...", total=len(images))

        for img_path in images:
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix(f'.{fmt}' if fmt else img_path.suffix)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            _process_single(img_path, out_path, method, gamma,
                          color_space, fmt, device, None, None)
            progress.advance(task)


if __name__ == '__main__':
    cli()
```

### 5. Metrics Module (`sece/metrics/`)

```python
# metrics/emeg.py
import numpy as np
import cv2

def emeg(image: np.ndarray, w1: int = 8, w2: int = 8,
         beta: float = 255.0, epsilon: float = 1e-10) -> float:
    """
    Expected Measure of Enhancement by Gradient.

    Formula (14) from paper. Measures contrast enhancement level.
    Higher EMEG = higher contrast.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    H, W = image.shape

    # Compute gradients
    dx = np.abs(np.diff(image, axis=1, prepend=image[:, 0:1]))
    dy = np.abs(np.diff(image, axis=0, prepend=image[0:1, :]))

    k1, k2 = H // w1, W // w2

    total = 0.0
    count = 0

    for i in range(k1):
        for j in range(k2):
            block_dx = dx[i*w1:(i+1)*w1, j*w2:(j+1)*w2]
            block_dy = dy[i*w1:(i+1)*w1, j*w2:(j+1)*w2]

            dx_h, dx_l = np.max(block_dx), np.min(block_dx)
            dy_h, dy_l = np.max(block_dy), np.min(block_dy)

            ratio_dx = dx_h / (dx_l + epsilon)
            ratio_dy = dy_h / (dy_l + epsilon)

            total += max(ratio_dx, ratio_dy) / beta
            count += 1

    return total / count if count > 0 else 0.0


# metrics/gmsd.py
def gmsd(image1: np.ndarray, image2: np.ndarray,
         c: float = 170.0) -> float:
    """
    Gradient Magnitude Similarity Deviation.

    Measures perceptual distortion between original and enhanced.
    Lower GMSD = less distortion.
    GMSD > 0.1 indicates visually noticeable distortion.
    """
    if image1.ndim == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if image2.ndim == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Sobel gradients
    def gradient_magnitude(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)

    G1 = gradient_magnitude(image1.astype(np.float64))
    G2 = gradient_magnitude(image2.astype(np.float64))

    # Gradient Magnitude Similarity
    GMS = (2 * G1 * G2 + c) / (G1**2 + G2**2 + c)

    # Standard deviation as final measure
    return float(np.std(GMS))


# metrics/ssim.py
def ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Structural Similarity Index.

    Wrapper around skimage.metrics.structural_similarity.
    """
    from skimage.metrics import structural_similarity as ssim_func

    if image1.ndim == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if image2.ndim == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    return float(ssim_func(image1, image2))
```

---

## Data Models

### Image Types

```python
from typing import TypeAlias, Union
import numpy as np

# Type aliases for clarity
GrayImage: TypeAlias = np.ndarray  # Shape: (H, W), dtype: uint8
ColorImage: TypeAlias = np.ndarray  # Shape: (H, W, 3), dtype: uint8
Image: TypeAlias = Union[GrayImage, ColorImage]

class EnhancementResult:
    """Container for enhancement results."""

    image: np.ndarray
    original: np.ndarray
    metrics: dict[str, float]
    method: str
    parameters: dict
```

---

## API Reference

### Public API (`sece/__init__.py`)

```python
"""SECE - Spatial Entropy-based Contrast Enhancement."""

from sece.core.sece import SECEEnhancer
from sece.core.secedct import SECEDCTEnhancer
from sece.color.processor import HSVProcessor, LABProcessor, YCbCrProcessor
from sece.metrics.emeg import emeg
from sece.metrics.gmsd import gmsd
from sece.metrics.ssim import ssim

__version__ = "0.1.0"
__all__ = [
    "SECEEnhancer",
    "SECEDCTEnhancer",
    "HSVProcessor",
    "LABProcessor",
    "YCbCrProcessor",
    "emeg",
    "gmsd",
    "ssim",
]
```

---

## Configuration

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sece"
version = "0.1.0"
description = "Spatial Entropy-based Contrast Enhancement"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "your@email.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Image Processing",
]
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
    "opencv-python>=4.5",
]

[project.optional-dependencies]
gpu = ["torch>=2.0"]
cli = ["click>=8.0", "rich>=13.0"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "mypy>=1.0",
    "ruff>=0.1.0",
]
docs = ["sphinx>=7.0", "furo>=2024.0"]
all = ["sece[gpu,cli,dev,docs]"]

[project.scripts]
sece = "sece.cli.main:cli"

[project.urls]
Homepage = "https://github.com/yourname/sece"
Documentation = "https://sece.readthedocs.io"
Repository = "https://github.com/yourname/sece"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=sece --cov-report=term-missing"

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
```

---

## File Organization

```
sece-rebuild/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── sece/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── sece.py
│       │   ├── secedct.py
│       │   └── base.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── numpy_backend.py
│       │   └── torch_backend.py
│       ├── color/
│       │   ├── __init__.py
│       │   └── processor.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── reader.py
│       │   └── writer.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── emeg.py
│       │   ├── gmsd.py
│       │   └── ssim.py
│       ├── baselines/
│       │   ├── __init__.py
│       │   ├── ghe.py
│       │   ├── wthe.py
│       │   └── clahe.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py
│       └── utils/
│           ├── __init__.py
│           └── validation.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_sece.py
│   ├── test_secedct.py
│   ├── test_metrics.py
│   └── test_cli.py
├── docs/
│   ├── conf.py
│   ├── index.md
│   └── api.md
└── examples/
    ├── basic_usage.py
    └── batch_processing.py
```

---

## Performance Targets

| Operation | NumPy Backend | Torch Backend (RTX 3050 Ti) |
|-----------|---------------|----------------------------|
| 512x512 SECE | < 1.0s | < 0.2s |
| 512x512 SECEDCT | < 1.5s | < 0.3s |
| 4K SECE | < 8s | < 2s |
| 4K SECEDCT | < 12s | < 3s |

---

## Testing Strategy

### Unit Tests
- Test each formula implementation against expected values
- Test edge cases (small images, single-color images)
- Test color space conversions

### Integration Tests
- Test full SECE pipeline
- Test full SECEDCT pipeline
- Test CLI commands

### Performance Tests
- Benchmark against target times
- Memory profiling for large images

---

## CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Lint with ruff
      run: ruff check src/

    - name: Type check with mypy
      run: mypy src/

    - name: Run tests
      run: pytest --cov=sece --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3

  publish:
    needs: test
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Build and publish
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

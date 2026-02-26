# SECE Test Datasets

This directory contains test images for SECE algorithm validation and evaluation.

## Directory Structure

```
data/
├── sample_images/     # Built-in test images from scikit-image (always available)
├── berkeley500/       # Berkeley Segmentation Dataset 500 (manual download)
└── csiq/              # CSIQ Image Quality Database (manual download)
```

## Sample Images (Built-in)

The `sample_images/` folder contains test images generated from scikit-image's built-in dataset. These are always available and require no external download.

| Image | Type | Size | Description |
|-------|------|------|-------------|
| `camera.png` | Grayscale | 512x512 | Classic cameraman test image |
| `coins.png` | Grayscale | 303x384 | Coins on textured background |
| `moon.png` | Grayscale | 512x512 | Lunar surface image |
| `page.png` | Grayscale | 191x191 | Scanned text document |
| `chelsea.png` | Color | 300x451 | Cat face (RGB test) |
| `astronaut.png` | Color | 512x512 | Astronaut portrait (RGB test) |
| `low_contrast.png` | Grayscale | 256x256 | Synthetic low contrast image |
| `high_contrast.png` | Grayscale | 256x256 | Synthetic high contrast image |
| `gradient.png` | Grayscale | 256x256 | Smooth gradient (edge case) |
| `checkerboard.png` | Grayscale | 256x256 | Binary pattern |

## External Datasets

### Berkeley Segmentation Dataset 500 (BSDS500)

**Source**: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/

**Alternative (Kaggle)**: https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500

**Download Instructions**:
```bash
# Option 1: Direct download (requires registration)
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS500.tgz
tar -xzf BSDS500.tgz -C berkeley500/

# Option 2: Kaggle CLI (requires kaggle account and API key)
kaggle datasets download -d balraj98/berkeley-segmentation-dataset-500-bsds500
unzip berkeley-segmentation-dataset-500-bsds500.zip -d berkeley500/
```

**Contents**: 500 natural images (200 train, 100 validation, 200 test)

### CSIQ Image Quality Database

**Source**: https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/

**Download Instructions**:
1. Visit the Qualinet database page
2. Register/login if required
3. Download the contrast decrement category images
4. Extract to `data/csiq/`

**Contents**: 30 reference images with 6 contrast levels each

## Generating Sample Images

To regenerate the sample images from scikit-image:

```python
from skimage import data, io, color
import numpy as np
from pathlib import Path

output_dir = Path("data/sample_images")
output_dir.mkdir(exist_ok=True)

# Standard grayscale images
io.imsave(output_dir / "camera.png", data.camera())
io.imsave(output_dir / "coins.png", data.coins())
io.imsave(output_dir / "moon.png", data.moon())
io.imsave(output_dir / "page.png", data.page())

# Color images
io.imsave(output_dir / "chelsea.png", data.chelsea())
io.imsave(output_dir / "astronaut.png", data.astronaut())

# Synthetic test images
np.random.seed(42)

# Low contrast: narrow histogram [100, 150]
low_contrast = np.random.randint(100, 151, (256, 256), dtype=np.uint8)
io.imsave(output_dir / "low_contrast.png", low_contrast)

# High contrast: full range [0, 255]
high_contrast = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
io.imsave(output_dir / "high_contrast.png", high_contrast)

# Smooth gradient
gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
io.imsave(output_dir / "gradient.png", gradient)

# Checkerboard pattern
checker = np.indices((256, 256)).sum(axis=0) % 2 * 255
checker = checker.astype(np.uint8)
io.imsave(output_dir / "checkerboard.png", checker)

print(f"Generated 10 sample images in {output_dir}")
```

## Usage in Tests

```python
from pathlib import Path
import cv2

DATA_DIR = Path(__file__).parent.parent / "data" / "sample_images"

def load_test_image(name: str) -> np.ndarray:
    """Load a test image by name."""
    path = DATA_DIR / name
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
```

## License

- **Sample images (from scikit-image)**: BSD 3-Clause license
- **BSDS500**: BSD license (free for research use)
- **CSIQ**: Free for research use with attribution

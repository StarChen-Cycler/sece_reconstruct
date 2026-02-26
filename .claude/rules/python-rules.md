# Python Coding Rules - SECE-Rebuild

## General Principles

1. **Type Hints Required**: All functions must have complete type annotations
2. **Docstrings**: NumPy-style docstrings for all public functions
3. **Line Length**: 88 characters (Black default)
4. **Import Order**: stdlib → third-party → local (isort)

## Code Style

### Naming Conventions

```python
# Classes: PascalCase
class SECEEnhancer: ...

# Functions/Methods: snake_case
def compute_spatial_histogram(): ...

# Constants: UPPER_SNAKE_CASE
DEFAULT_GAMMA = 0.5

# Private methods: _leading_underscore
def _compute_grid_size(): ...

# Type aliases: PascalCase
GrayImage: TypeAlias = np.ndarray
```

### Function Signatures

```python
def enhance(
    self,
    image: np.ndarray,
    gamma: float = 0.5,
    *,
    color_space: str = 'hsv',
    backend: str = 'numpy'
) -> np.ndarray:
    """
    Enhance image contrast.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W) or (H, W, 3)
    gamma : float, optional
        Local enhancement level [0, 1], by default 0.5
    color_space : str, optional
        Color space for processing, by default 'hsv'
    backend : str, optional
        Backend ('numpy' or 'torch'), by default 'numpy'

    Returns
    -------
    np.ndarray
        Enhanced image with same shape as input

    Raises
    ------
    ValueError
        If image shape is invalid or gamma out of range
    """
```

## NumPy Best Practices

### Array Operations

```python
# GOOD: Vectorized operations
result = np.sum(region == x_k)

# BAD: Python loops for array operations
# total = 0
# for pixel in region.flat:
#     if pixel == x_k:
#         total += 1

# GOOD: Use built-in functions
levels = np.unique(image)

# GOOD: Avoid copies when possible
Y = np.zeros_like(image, dtype=np.uint8)  # Reuse shape
```

### Type Handling

```python
# Always specify dtype for numerical arrays
h_k = np.zeros((M, N), dtype=np.float64)

# Use uint8 for images
image_uint8 = image.astype(np.uint8)

# Use float64 for computations
image_float = image.astype(np.float64)
```

## Error Handling

### Raw Error Propagation

```python
# GOOD: Let errors propagate for debugging
def enhance(self, image: np.ndarray) -> np.ndarray:
    # Don't wrap - actual error surfaces
    return self._process(image)

# Only catch when adding context
def load_image(self, path: str) -> np.ndarray:
    try:
        return cv2.imread(path)
    except Exception as e:
        # Add context but preserve original error
        raise ImageLoadError(f"Failed to load {path}: {e}") from e
```

### Custom Exceptions

```python
class SECEError(Exception):
    """Base exception for SECE library."""
    pass

class UnsupportedColorSpaceError(SECEError):
    """Color space not supported."""
    pass

class InvalidParameterError(SECEError):
    """Invalid parameter value."""
    pass
```

## Performance Rules

### Memory Efficiency

```python
# GOOD: In-place operations when safe
np.clip(result, 0, 255, out=result)

# GOOD: Pre-allocate output
output = np.empty_like(input)

# BAD: Repeated allocation in loops
# for i in range(n):
#     result = np.zeros(shape)  # Allocates each iteration
```

### Avoid Premature Optimization

1. Write clear, correct code first
2. Profile before optimizing
3. Use vectorized NumPy operations
4. Consider PyTorch for GPU acceleration

## Testing Rules

### Test Organization

```python
# tests/test_sece.py
import pytest
import numpy as np

class TestSECEEnhancer:
    """Tests for SECEEnhancer class."""

    @pytest.fixture
    def enhancer(self):
        return SECEEnhancer()

    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    def test_enhance_returns_same_shape(self, enhancer, sample_image):
        result = enhancer.enhance(sample_image)
        assert result.shape == sample_image.shape

    def test_enhance_uses_full_dynamic_range(self, enhancer, sample_image):
        result = enhancer.enhance(sample_image)
        assert result.min() >= 0
        assert result.max() <= 255
```

### Edge Case Testing

```python
@pytest.mark.parametrize("size,expected_warning", [
    ((8, 8), "small image"),
    ((16, 16), "small image"),
    ((32, 32), None),
])
def test_small_image_warning(size, expected_warning):
    image = np.random.randint(0, 256, size, dtype=np.uint8)
    # Test warning behavior
```

## Documentation Rules

### Inline Comments

```python
# Only comment non-obvious logic
# Formula (2): Grid size preserves image aspect ratio
r = H / W
N = int(np.floor(np.sqrt(K / r)))
M = int(np.floor(np.sqrt(K * r)))

# No comments for obvious code
result = image + 1  # BAD: obvious
```

### Module Docstrings

```python
"""SECE Core Algorithm Module.

Implements the Spatial Entropy-based Contrast Enhancement (SECE)
algorithm from Celik (2014).

Key Components:
- SECEEnhancer: Global contrast enhancement
- SECEDCTEnhancer: Global + local contrast enhancement

Reference:
    T. Celik, "Spatial Entropy-Based Global and Local Image Contrast
    Enhancement," IEEE Trans. Image Process., 2014.
"""
```

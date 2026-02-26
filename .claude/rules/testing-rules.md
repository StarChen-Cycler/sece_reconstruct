# Testing Rules - SECE-Rebuild

## Testing Framework

- **Framework**: pytest >= 7.0
- **Coverage**: pytest-cov >= 4.0
- **Target Coverage**: > 80%

## Test Organization

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── test_sece.py          # SECE algorithm tests
├── test_secedct.py       # SECEDCT algorithm tests
├── test_backends.py      # Backend tests
├── test_color.py         # Color processing tests
├── test_metrics.py       # Metrics tests
├── test_baselines.py     # Baseline algorithm tests
├── test_cli.py           # CLI tests
└── fixtures/
    └── test_images/      # Test images
```

## Fixture Patterns

### conftest.py

```python
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_grayscale():
    """Standard 128x128 grayscale test image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (128, 128), dtype=np.uint8)

@pytest.fixture
def sample_color():
    """Standard 128x128 color test image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

@pytest.fixture
def low_contrast_image():
    """Low contrast test image (narrow histogram)."""
    np.random.seed(42)
    return np.random.randint(100, 150, (128, 128), dtype=np.uint8)

@pytest.fixture
def single_color_image():
    """Uniform color image for edge case testing."""
    return np.full((64, 64), 128, dtype=np.uint8)

@pytest.fixture
def small_image():
    """Tiny image for edge case testing."""
    return np.random.randint(0, 256, (8, 8), dtype=np.uint8)

@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"
```

## Test Patterns

### Algorithm Correctness Tests

```python
class TestSECEFormulas:
    """Test SECE formula implementations."""

    def test_grid_size_preserves_aspect_ratio(self):
        """Formula (2): M/N should equal H/W."""
        K = 100
        H, W = 480, 640
        M, N = compute_grid_size(K, (H, W))
        assert abs(M/N - H/W) < 0.1  # Allow rounding

    def test_spatial_entropy_range(self, sample_grayscale):
        """Formula (3): Entropy should be >= 0."""
        h_k = compute_spatial_histogram(sample_grayscale, 128, 10, 10)
        entropy = compute_spatial_entropy(h_k)
        assert entropy >= 0

    def test_cdf_monotonic(self, sample_grayscale):
        """Formula (6): CDF should be monotonically increasing."""
        _, _, F = compute_distribution(sample_grayscale)
        assert np.all(np.diff(F) >= 0)

    def test_mapping_range(self):
        """Formula (7): Output should be in [y_d, y_u]."""
        F = np.linspace(0, 1, 256)
        y = mapping_function(F, y_d=0, y_u=255)
        assert y.min() >= 0
        assert y.max() <= 255
```

### Integration Tests

```python
class TestSECEIntegration:
    """Integration tests for full SECE pipeline."""

    def test_sece_returns_same_shape(self, sample_grayscale):
        enhancer = SECEEnhancer()
        result = enhancer.enhance(sample_grayscale)
        assert result.shape == sample_grayscale.shape

    def test_sece_uses_full_dynamic_range(self, low_contrast_image):
        enhancer = SECEEnhancer()
        result = enhancer.enhance(low_contrast_image)
        # Should expand to use more of the range
        assert result.max() - result.min() > low_contrast_image.max() - low_contrast_image.min()

    def test_secedct_gamma_zero_equals_sece(self, sample_grayscale):
        """gamma=0 should produce identical results to SECE."""
        sece = SECEEnhancer()
        secedct = SECEDCTEnhancer(gamma=0)

        result_sece = sece.enhance(sample_grayscale)
        result_secedct = secedct.enhance(sample_grayscale)

        np.testing.assert_array_equal(result_sece, result_secedct)

    def test_dct_roundtrip(self, sample_grayscale):
        """DCT -> IDCT should recover original."""
        backend = NumpyBackend()
        D = backend.dct2d(sample_grayscale.astype(np.float64))
        recovered = backend.idct2d(D)
        np.testing.assert_allclose(recovered, sample_grayscale, rtol=1e-10)
```

### Edge Case Tests

```python
class TestEdgeCases:
    """Edge case handling tests."""

    def test_single_color_image(self, single_color_image):
        """Single color should return unchanged."""
        enhancer = SECEEnhancer()
        result = enhancer.enhance(single_color_image)
        np.testing.assert_array_equal(result, single_color_image)

    def test_small_image_warning(self, small_image):
        """Small images should emit warning."""
        enhancer = SECEEnhancer()
        with pytest.warns(UserWarning, match="small image"):
            enhancer.enhance(small_image)

    def test_rgba_image_handling(self):
        """RGBA should be converted to RGB."""
        rgba = np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)
        enhancer = SECEEnhancer()
        with pytest.warns(UserWarning, match="alpha"):
            result = enhancer.enhance(rgba)
        assert result.ndim == 3
        assert result.shape[2] == 3

    @pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
    def test_non_uint8_conversion(self, dtype):
        """Non-uint8 images should be converted with warning."""
        image = np.random.randint(0, 65536, (64, 64)).astype(dtype)
        enhancer = SECEEnhancer()
        with pytest.warns(UserWarning, match="8-bit"):
            result = enhancer.enhance(image)
        assert result.dtype == np.uint8
```

### Performance Tests

```python
import pytest
import time

class TestPerformance:
    """Performance benchmark tests."""

    @pytest.fixture
    def image_512(self):
        return np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    def test_sece_performance_512(self, image_512):
        """SECE should process 512x512 in < 1 second."""
        enhancer = SECEEnhancer()
        start = time.perf_counter()
        enhancer.enhance(image_512)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"SECE took {elapsed:.2f}s (expected < 1s)"

    def test_secedct_performance_512(self, image_512):
        """SECEDCT should process 512x512 in < 1.5 seconds."""
        enhancer = SECEDCTEnhancer()
        start = time.perf_counter()
        enhancer.enhance(image_512)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.5, f"SECEDCT took {elapsed:.2f}s (expected < 1.5s)"
```

### Metric Tests

```python
class TestMetrics:
    """Quality metric tests."""

    def test_emeg_identical_images(self, sample_grayscale):
        """EMEG(X, X) should give same value."""
        e1 = emeg(sample_grayscale)
        e2 = emeg(sample_grayscale)
        assert e1 == e2

    def test_emeg_higher_after_enhancement(self, low_contrast_image):
        """Enhanced image should have higher EMEG."""
        enhancer = SECEDCTEnhancer()
        enhanced = enhancer.enhance(low_contrast_image)

        e_before = emeg(low_contrast_image)
        e_after = emeg(enhanced)

        assert e_after > e_before, "EMEG should increase after enhancement"

    def test_gmsd_identical_images(self, sample_grayscale):
        """GMSD(X, X) should be 0."""
        g = gmsd(sample_grayscale, sample_grayscale)
        assert g == 0.0

    def test_gmsd_different_images(self, sample_grayscale, low_contrast_image):
        """GMSD(X, Y) should be > 0 when X != Y."""
        g = gmsd(sample_grayscale, low_contrast_image)
        assert g > 0.0
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sece --cov-report=html

# Run specific test file
pytest tests/test_sece.py

# Run specific test class
pytest tests/test_sece.py::TestSECEFormulas

# Run with verbose output
pytest -v

# Run only fast tests
pytest -m "not slow"

# Run performance benchmarks
pytest -m "benchmark"
```

## Coverage Requirements

| Module | Minimum Coverage |
|--------|------------------|
| sece.core | 90% |
| sece.backends | 85% |
| sece.color | 85% |
| sece.metrics | 90% |
| sece.cli | 70% |
| **Overall** | **80%** |

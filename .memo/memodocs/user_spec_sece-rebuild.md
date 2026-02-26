# User Specification: SECE-Rebuild

**Project**: sece-rebuild
**Type**: Python Library (PyPI Package)
**Paper Reference**: Turgay Celik, "Spatial Entropy-Based Global and Local Image Contrast Enhancement," IEEE Trans. Image Process., 2014

---

## Executive Summary

SECE-Rebuild is a production-ready Python library implementing the SECE (Spatial Entropy-based Contrast Enhancement) and SECEDCT algorithms from Turgay Celik's 2014 IEEE paper. The library provides artifact-free global and local contrast enhancement for grayscale and color images with a clean, extensible class-based API.

---

## Core Requirements

### 1. Primary Features

| Feature | Description | Priority |
|---------|-------------|----------|
| SECE Algorithm | Global contrast enhancement using spatial entropy | P0 |
| SECEDCT Algorithm | Global + local contrast enhancement via DCT weighting | P0 |
| Color Image Support | HSV, LAB, YCbCr color space processing | P0 |
| CLI Tool | Batch/single image processing with format detection | P1 |
| Quality Metrics | EMEG, GMSD, SSIM evaluation metrics | P1 |
| Baseline Algorithms | GHE, WTHE, CLAHE for benchmarking | P1 |
| Dual Backend | NumPy (CPU) + PyTorch (GPU) support | P2 |

### 2. API Design

**Class-Based Architecture**:
```python
from sece import SECEEnhancer, SECEDCTEnhancer

# Initialize enhancer
enhancer = SECEEnhancer(color_space='hsv')
enhancer_dct = SECEDCTEnhancer(gamma=0.5, color_space='lab')

# Process images
result = enhancer.enhance(image)
result, metrics = enhancer_dct.enhance_with_metrics(image)

# Batch processing
results = enhancer.batch_process(['img1.png', 'img2.jpg'])
```

**Complete Image I/O**:
- Automatic format detection (PNG, JPEG, TIFF, BMP, WEBP)
- Support for grayscale and color images
- Automatic alpha channel handling (RGBA → RGB conversion)
- Output path and format selection

### 3. CLI Tool Requirements

```bash
# Single image
sece input.png -o output.png --method secedct --gamma 0.5

# Batch processing
sece ./input_folder/ -o ./output_folder/ --format png --method sece

# With metrics
sece input.png -o output.png --metrics emeg,gmsd,ssim

# Compare with baseline
sece input.png -o output.png --compare ghe,wthe

# GPU acceleration
sece input.png -o output.png --device cuda
```

**CLI Features**:
- Single image or folder batch processing
- Automatic format detection and selection
- Output path and format configuration
- Method selection (sece/secedct)
- Gamma parameter for SECEDCT
- Metrics calculation
- Baseline comparison
- Device selection (cpu/cuda)

---

## User Stories

### US-1: Basic Enhancement
> As a researcher, I want to enhance a single image using SECE so that I can improve its contrast without artifacts.

**Acceptance Criteria**:
- Load image from file path
- Apply SECE enhancement
- Save to specified output path
- Processing time < 1s for 512x512 image

### US-2: Batch Processing
> As a data scientist, I want to process an entire folder of images so that I can enhance my dataset efficiently.

**Acceptance Criteria**:
- Process all images in folder recursively
- Preserve folder structure in output
- Support format conversion
- Report progress and any errors

### US-3: Quality Evaluation
> As a researcher, I want to measure EMEG and GMSD scores so that I can quantify the enhancement quality.

**Acceptance Criteria**:
- Calculate EMEG for input and output
- Calculate GMSD between input and output
- Return metrics in structured format
- Compare with baseline algorithms

### US-4: GPU Acceleration
> As a user with large images, I want to use GPU acceleration so that processing is faster.

**Acceptance Criteria**:
- Auto-detect CUDA availability
- Fall back to CPU if GPU unavailable
- Process 4K images in < 2s on GPU
- Memory-efficient for 4GB VRAM

---

## Technical Specifications

### Dependencies

**Core**:
- numpy >= 1.20
- scipy >= 1.7 (fftpack for DCT)
- opencv-python >= 4.5

**Optional (GPU)**:
- torch >= 2.0 (optional, for GPU backend)

**CLI**:
- click >= 8.0
- rich >= 13.0 (progress bars)

**Development**:
- pytest >= 7.0
- pytest-cov >= 4.0
- black >= 23.0
- mypy >= 1.0

### Module Structure

```
sece/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── sece.py           # SECE algorithm
│   ├── secedct.py        # SECEDCT algorithm
│   ├── spatial_entropy.py
│   ├── distribution.py
│   └── mapping.py
├── transforms/
│   ├── __init__.py
│   ├── dct.py            # 2D-DCT transforms
│   └── weighting.py      # Coefficient weighting
├── color/
│   ├── __init__.py
│   ├── hsv.py
│   ├── lab.py
│   └── ycbcr.py
├── io/
│   ├── __init__.py
│   ├── reader.py         # Image loading
│   └── writer.py         # Image saving
├── metrics/
│   ├── __init__.py
│   ├── emeg.py
│   ├── gmsd.py
│   └── ssim.py
├── baselines/
│   ├── __init__.py
│   ├── ghe.py
│   ├── wthe.py
│   └── clahe.py
├── backends/
│   ├── __init__.py
│   ├── numpy_backend.py
│   └── torch_backend.py
├── cli/
│   ├── __init__.py
│   └── main.py
└── utils/
    ├── __init__.py
    └── validation.py
```

### Design Patterns

**Strategy Pattern** for backends:
```python
class Backend(ABC):
    @abstractmethod
    def dct2d(self, x: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def idct2d(self, x: ArrayLike) -> ArrayLike: ...

class NumpyBackend(Backend): ...
class TorchBackend(Backend): ...
```

**Factory Pattern** for enhancers:
```python
class EnhancerFactory:
    @staticmethod
    def create(method: str, **kwargs) -> BaseEnhancer:
        if method == 'sece':
            return SECEEnhancer(**kwargs)
        elif method == 'secedct':
            return SECEDCTEnhancer(**kwargs)
```

**Template Method** for color processing:
```python
class ColorProcessor(ABC):
    @abstractmethod
    def to_luminance(self, image): ...

    @abstractmethod
    def from_luminance(self, image, luminance): ...
```

---

## Edge Cases & Error Handling

### Edge Cases to Handle

| Case | Behavior |
|------|----------|
| Small images (<32x32) | Warning, process with adjusted grid size |
| Non-8bit images (16-bit, float32) | Auto-convert to 8-bit with scaling |
| RGBA images | Strip alpha channel with warning |
| Low dynamic range (<16 gray levels) | Warning, may produce limited enhancement |
| Single-color images | Return unchanged with warning |
| Very large images (>4K) | Memory warning, suggest GPU or tiling |

### Error Strategy

**Raw Error Feedback** - Expose actual errors for debugging:

```python
# Don't wrap errors - let them propagate
def enhance(self, image):
    # Direct error propagation, not wrapped
    return self._process(image)  # ValueError, TypeError pass through

# Custom exceptions only for library-specific errors
class SECEError(Exception): ...
class UnsupportedColorSpaceError(SECEError): ...
class InvalidParameterError(SECEError): ...
```

---

## Success Criteria

### Functional Requirements

- [ ] SECE produces artifact-free global enhancement
- [ ] SECEDCT provides controllable local enhancement (gamma parameter)
- [ ] Output uses full dynamic range [0, 255]
- [ ] Histogram shape preserved (SECE property)
- [ ] Color images enhanced correctly in multiple color spaces

### Performance Requirements

| Metric | Target |
|--------|--------|
| 512x512 image (CPU) | < 1 second |
| 512x512 image (GPU) | < 0.2 seconds |
| 4K image (GPU) | < 2 seconds |
| Memory usage (4GB VRAM) | Fit without OOM |

### Quality Requirements

- [ ] EMEG(Y) > EMEG(X) for low-contrast images
- [ ] GMSD(X, Y) < 0.1 (minimal distortion threshold)
- [ ] Face recognition improvement on CMU PIE database
- [ ] Test coverage > 80%

---

## Iterative Growth Plan

### Phase 1: Core (MVP)
- SECE algorithm implementation
- SECEDCT algorithm implementation
- Grayscale image support
- Basic CLI (single image)
- Unit tests

### Phase 2: Color & I/O
- HSV color space support
- LAB, YCbCr color spaces
- Complete image I/O
- Batch CLI processing
- Integration tests

### Phase 3: Metrics & Baselines
- EMEG metric
- GMSD metric
- SSIM metric
- GHE, WTHE, CLAHE baselines
- Comparison utilities

### Phase 4: GPU & Advanced
- PyTorch backend
- GPU acceleration
- Auto dataset download (Berkeley subset)
- Performance optimization

### Phase 5: Polish & Release
- Full documentation (Sphinx)
- PyPI package
- CI/CD pipeline
- ReadTheDocs hosting

---

## Hardware Considerations

**Target Development Environment**:
- GPU: NVIDIA RTX 3050 Ti (4GB VRAM)
- CPU: AMD Ryzen 7 5800H (8 cores / 16 threads)
- RAM: 64 GB

**Memory Optimization for 4GB VRAM**:
- Mixed precision processing
- Batch size limits for GPU
- Gradient checkpointing not needed (inference only)
- Fallback to CPU for large images

---

## Dataset Handling

**Automatic Dataset Download**:
- Berkeley 500 subset (10-20 representative images)
- CSIQ contrast decrement images
- Store in `~/.sece/datasets/`
- Download on first use with progress bar

**Dataset Sources**:
- BSDS500: Kaggle (kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)
- CSIQ: Qualinet (qualinet.github.io/databases/image/categorical_image_quality_csiq_database/)

---

## Non-Goals (Out of Scope)

- Real-time video processing
- GUI application
- Web service/API
- Mobile deployment
- Custom neural network enhancement
- RAW image format support

---

## References

- Paper: T. Celik, "Spatial Entropy-Based Global and Local Image Contrast Enhancement," IEEE TIP 2014
- DOI: 10.1109/TIP.2014.2364537
- Reference Implementation: celikturgay@gmail.com (email with SECEDCT subject)
- Similar Project: github.com/xiezhongzhao/Contrast_Enhancement (26 stars)

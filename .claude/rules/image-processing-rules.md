# Image Processing Rules - SECE-Rebuild

## Core Algorithm Rules

### Formula Implementation

All formulas must match the paper exactly:

| Formula | Description | Implementation Check |
|---------|-------------|---------------------|
| (1) | 2D spatial histogram | h_k[m,n] counts pixels == x_k in grid |
| (2) | Grid size | M = floor(sqrt(K*r)), N = floor(sqrt(K/r)) |
| (3) | Spatial entropy | S_k = -sum(h_k * log2(h_k)) |
| (4) | Discrete function | f_k = S_k / sum(S_l where l != k) |
| (5) | Normalization | f_k = f_k / sum(f) |
| (6) | CDF | F_k = cumsum(f) |
| (7) | Mapping | y_k = floor(F_k * (y_u - y_d) + y_d) |
| (10-11) | DCT weighting | w(k,l) uses alpha parameter |
| (12) | Alpha calculation | alpha = entropy(f)^gamma |
| (14) | EMEG | Block-based gradient measure |

### Numerical Stability

```python
# Always add epsilon to prevent log(0)
epsilon = 1e-10
entropy = -np.sum(p * np.log2(p + epsilon))

# Use np.clip to prevent overflow
result = np.clip(result, 0, 255)

# Handle division by zero
if denominator > epsilon:
    result = numerator / denominator
else:
    result = 0.0
```

### Grid Size Constraints

```python
def _compute_grid_size(K: int, H: int, W: int) -> Tuple[int, int]:
    """
    Compute grid size with minimum constraints.

    Rules:
    - M, N >= 1 (minimum 1x1 grid)
    - Preserve aspect ratio as much as possible
    - Total grids MN should be close to K
    """
    r = H / W
    N = max(1, int(np.floor(np.sqrt(K / r))))
    M = max(1, int(np.floor(np.sqrt(K * r))))
    return M, N
```

## Color Processing Rules

### Color Space Handling

```python
# Always convert BGR to target space before processing
def to_luminance(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract luminance channel.

    Input: BGR image from cv2.imread()
    Output: (luminance, chrominance) tuple
    """
    pass

# Restore BGR format for cv2.imwrite()
def from_luminance(self, luminance: np.ndarray, chrominance: np.ndarray) -> np.ndarray:
    """
    Combine luminance with chrominance.

    Output: BGR image for cv2.imwrite()
    """
    pass
```

### Color Space Priority

1. **HSV** (default): Perceptually uniform luminance (V channel)
2. **LAB**: Device-independent, good for scientific applications
3. **YCbCr**: Video standard, good for JPEG-like processing

## Image I/O Rules

### Format Handling

```python
SUPPORTED_FORMATS = {
    '.png': cv2.IMREAD_UNCHANGED,
    '.jpg': cv2.IMREAD_COLOR,
    '.jpeg': cv2.IMREAD_COLOR,
    '.tif': cv2.IMREAD_UNCHANGED,
    '.tiff': cv2.IMREAD_UNCHANGED,
    '.bmp': cv2.IMREAD_COLOR,
    '.webp': cv2.IMREAD_COLOR,
}

def load_image(path: str) -> np.ndarray:
    """Load image with automatic format detection."""
    ext = Path(path).suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(f"Unsupported format: {ext}")

    flags = SUPPORTED_FORMATS[ext]
    image = cv2.imread(path, flags)

    if image is None:
        raise ImageLoadError(f"Failed to load: {path}")

    return image
```

### Alpha Channel Handling

```python
def handle_alpha(image: np.ndarray) -> np.ndarray:
    """
    Handle RGBA images by stripping alpha channel.

    Warning: User should be notified of alpha removal.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        # Emit warning
        warnings.warn(
            "Alpha channel detected and removed. "
            "Transparency information will be lost.",
            UserWarning
        )
        return image[:, :, :3]  # BGR only
    return image
```

### Bit Depth Handling

```python
def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image to 8-bit with appropriate scaling.

    Rules:
    - uint8: return as-is
    - uint16: scale to 0-255
    - float [0,1]: scale to 0-255
    - float [0,255]: convert to uint8
    """
    if image.dtype == np.uint8:
        return image

    warnings.warn(f"Converting {image.dtype} to uint8", UserWarning)

    if image.dtype == np.uint16:
        return (image / 256).astype(np.uint8)

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        return image.astype(np.uint8)

    raise ValueError(f"Unsupported dtype: {image.dtype}")
```

## DCT Transform Rules

### Backend Selection

```python
def get_backend(name: str, device: str = 'cpu') -> Backend:
    """
    Get appropriate backend.

    Rules:
    - 'numpy' always available (CPU)
    - 'torch' requires PyTorch installation
    - 'cuda' requires PyTorch + CUDA GPU
    """
    if name == 'numpy':
        return NumpyBackend()

    if name == 'torch' or device == 'cuda':
        try:
            import torch
            if device == 'cuda' and not torch.cuda.is_available():
                warnings.warn("CUDA not available, falling back to CPU")
                device = 'cpu'
            return TorchBackend(device=device)
        except ImportError:
            warnings.warn("PyTorch not installed, falling back to NumPy")
            return NumpyBackend()

    raise ValueError(f"Unknown backend: {name}")
```

### DCT Accuracy

```python
def test_dct_roundtrip():
    """DCT -> IDCT must recover original with high precision."""
    x = np.random.randn(100, 100)
    D = dct2d(x)
    recovered = idct2d(D)
    mse = np.mean((x - recovered) ** 2)
    assert mse < 1e-10, f"DCT roundtrip MSE too high: {mse}"
```

## Edge Case Handling

### Image Size Limits

| Size | Behavior |
|------|----------|
| < 32x32 | Warning: small image, may have limited enhancement |
| < 8x8 | Warning: very small image, grid size adjusted |
| > 4096x4096 | Warning: large image, consider GPU or tiling |
| > 8192x8192 | Error: image too large, may cause OOM |

### Gray Level Limits

| Case | Behavior |
|------|----------|
| K = 1 | Return unchanged (single color) |
| K < 16 | Warning: low dynamic range |
| K >= 256 | Normal processing |

## Performance Guidelines

### Memory Management

```python
# For large images, use chunked processing
def process_large_image(image: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    """
    Process large images in chunks to avoid OOM.

    Note: This may introduce boundary artifacts.
    For best results, use GPU with sufficient VRAM.
    """
    if image.shape[0] * image.shape[1] > chunk_size * chunk_size:
        warnings.warn(
            f"Large image {image.shape}. Consider GPU processing.",
            UserWarning
        )
    # Process in chunks...
```

### GPU Memory (4GB VRAM)

```python
# Check available VRAM before processing
def estimate_vram_needed(H: int, W: int) -> int:
    """Estimate VRAM needed for image processing in bytes."""
    # DCT needs float64, multiple intermediate arrays
    return H * W * 8 * 5  # Rough estimate: 5x image size in float64

def check_gpu_memory(image_shape: Tuple[int, int]) -> bool:
    """Check if image fits in GPU memory."""
    try:
        import torch
        needed = estimate_vram_needed(*image_shape)
        available = torch.cuda.get_device_properties(0).total_memory
        return needed < available * 0.8  # Leave 20% buffer
    except:
        return False
```

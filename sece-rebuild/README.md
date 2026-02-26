# SECE - Spatial Entropy-based Contrast Enhancement

Python implementation of SECE (Spatial Entropy-based Contrast Enhancement) and SECEDCT algorithms from Turgay Celik's 2014 IEEE paper for artifact-free global and local image contrast enhancement.

## Installation

```bash
pip install sece
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from sece import SECEEnhancer, SECEDCTEnhancer

# Global contrast enhancement
enhancer = SECEEnhancer()
enhanced = enhancer.enhance(image)

# Global + local contrast enhancement
enhancer = SECEDCTEnhancer(gamma=0.5)
enhanced = enhancer.enhance(image)
```

## CLI

```bash
sece enhance input.png output.png --gamma 0.5
```

## Reference

T. Celik, "Spatial Entropy-Based Global and Local Image Contrast Enhancement," IEEE Trans. Image Process., vol. 23, no. 5, pp. 2148-2158, May 2014. DOI: 10.1109/TIP.2014.2364537

## License

MIT License

# SECE-Rebuild

**Git Root**: `I:\ai-automation-projects\research-paper-reconstruct`
**Working Directory**: `sece-rebuild/`

## Project Overview

Python implementation of SECE (Spatial Entropy-based Contrast Enhancement) and SECEDCT algorithms from Turgay Celik's 2014 IEEE paper. Production-ready PyPI package for artifact-free global and local image contrast enhancement.

## Directory Structure

```
research-paper-reconstruct/
├── .claude/                 # AI coding rules
│   └── rules/
│       ├── python-rules.md
│       ├── testing-rules.md
│       └── image-processing-rules.md
├── .memo/                   # Project documentation
│   └── memodocs/
│       ├── user_spec_sece-rebuild.md
│       └── tech_spec_sece-rebuild.md
├── sece-rebuild/            # Source code (create this)
│   ├── src/sece/
│   ├── tests/
│   └── pyproject.toml
└── CLAUDE.md                # This file
```

## Development Context

### Tech Stack
- Python >= 3.10
- NumPy >= 1.20 (core)
- SciPy >= 1.7 (DCT transforms)
- OpenCV >= 4.5 (image I/O)
- PyTorch >= 2.0 (optional GPU backend)
- Click >= 8.0 (CLI)
- pytest >= 7.0 (testing)

### Hardware
| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3050 Ti (4GB VRAM) |
| CPU | AMD Ryzen 7 5800H (8 cores / 16 threads) |
| RAM | 64 GB |

### Conda Environment
```bash
conda activate chatterbox  # PyTorch 2.4.1+cu124, CUDA 12.4
```

## Commands

```bash
# Navigate to source directory
cd I:/ai-automation-projects/research-paper-reconstruct/sece-rebuild

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=sece

# Format code
black src/
ruff check src/

# Type check
mypy src/

# Build docs
cd docs && make html

# Run CLI
sece --help
```

## Key References

- **Paper**: T. Celik, "Spatial Entropy-Based Global and Local Image Contrast Enhancement," IEEE TIP 2014
- **DOI**: 10.1109/TIP.2014.2364537
- **Reference Implementation**: celikturgay@gmail.com (email with SECEDCT subject)

## Octie Tasks

See `octie list --format md` for current task status. 16 tasks defined across 5 phases:
1. Foundation (setup, datasets)
2. SECE Core (spatial entropy, distribution, mapping)
3. SECEDCT (DCT transforms, weighting)
4. Evaluation (metrics, color support)
5. Testing & Documentation

## Next Steps

1. Use `/octie-research` to initialize Octie project and create tasks
2. Use `/octie-dev` for Phase 2 implementation loop

"""Shared pytest fixtures for SECE tests."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def sample_grayscale() -> np.ndarray:
    """Standard 128x128 grayscale test image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (128, 128), dtype=np.uint8)


@pytest.fixture
def sample_color() -> np.ndarray:
    """Standard 128x128 color test image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def low_contrast_image() -> np.ndarray:
    """Low contrast test image (narrow histogram)."""
    np.random.seed(42)
    return np.random.randint(100, 150, (128, 128), dtype=np.uint8)


@pytest.fixture
def single_color_image() -> np.ndarray:
    """Uniform color image for edge case testing."""
    return np.full((64, 64), 128, dtype=np.uint8)


@pytest.fixture
def small_image() -> np.ndarray:
    """Tiny image for edge case testing."""
    np.random.seed(42)
    return np.random.randint(0, 256, (8, 8), dtype=np.uint8)


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"

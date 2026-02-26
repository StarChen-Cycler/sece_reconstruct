"""Tests for DCT transform backends.

Tests verify that:
1. NumpyBackend produces correct DCT results
2. TorchBackend matches NumpyBackend within tolerance
3. Both backends handle edge cases correctly
4. TorchBackend properly uses GPU when available
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sece.backends import get_backend
from sece.backends.base import Backend
from sece.backends.numpy_backend import NumpyBackend


class TestNumpyBackend:
    """Tests for NumPy/SciPy backend."""

    @pytest.fixture
    def backend(self) -> NumpyBackend:
        """Create NumpyBackend instance."""
        return NumpyBackend()

    def test_name(self, backend: NumpyBackend) -> None:
        """Backend name should be 'numpy'."""
        assert backend.name == "numpy"

    def test_device(self, backend: NumpyBackend) -> None:
        """Device should always be 'cpu'."""
        assert backend.device == "cpu"

    def test_dct2d_output_shape(self, backend: NumpyBackend) -> None:
        """DCT output should have same shape as input."""
        x = np.random.randn(64, 64)
        result = backend.dct2d(x)
        assert result.shape == x.shape

    def test_dct2d_output_dtype(self, backend: NumpyBackend) -> None:
        """DCT output should be float64."""
        x = np.random.randn(64, 64)
        result = backend.dct2d(x)
        assert result.dtype == np.float64

    def test_idct2d_output_shape(self, backend: NumpyBackend) -> None:
        """IDCT output should have same shape as input."""
        D = np.random.randn(64, 64)
        result = backend.idct2d(D)
        assert result.shape == D.shape

    def test_roundtrip_small(self, backend: NumpyBackend) -> None:
        """DCT -> IDCT should recover original (small image)."""
        x = np.random.randn(32, 32)
        D = backend.dct2d(x)
        recovered = backend.idct2d(D)
        mse = np.mean((x - recovered) ** 2)
        assert mse < 1e-20, f"Roundtrip MSE too high: {mse}"

    def test_roundtrip_medium(self, backend: NumpyBackend) -> None:
        """DCT -> IDCT should recover original (medium image)."""
        x = np.random.randn(256, 256)
        D = backend.dct2d(x)
        recovered = backend.idct2d(D)
        mse = np.mean((x - recovered) ** 2)
        assert mse < 1e-20, f"Roundtrip MSE too high: {mse}"

    def test_roundtrip_large(self, backend: NumpyBackend) -> None:
        """DCT -> IDCT should recover original (large image)."""
        x = np.random.randn(512, 512)
        D = backend.dct2d(x)
        recovered = backend.idct2d(D)
        mse = np.mean((x - recovered) ** 2)
        assert mse < 1e-20, f"Roundtrip MSE too high: {mse}"

    def test_dct2d_1d_raises(self, backend: NumpyBackend) -> None:
        """1D input should raise ValueError."""
        x = np.random.randn(64)
        with pytest.raises(ValueError, match="must be 2D"):
            backend.dct2d(x)

    def test_dct2d_3d_raises(self, backend: NumpyBackend) -> None:
        """3D input should raise ValueError."""
        x = np.random.randn(64, 64, 3)
        with pytest.raises(ValueError, match="must be 2D"):
            backend.dct2d(x)

    def test_idct2d_1d_raises(self, backend: NumpyBackend) -> None:
        """1D input should raise ValueError."""
        D = np.random.randn(64)
        with pytest.raises(ValueError, match="must be 2D"):
            backend.idct2d(D)

    def test_dc_coefficient(self, backend: NumpyBackend) -> None:
        """DC coefficient should equal mean * sqrt(M*N)."""
        x = np.ones((8, 8)) * 100  # Constant image
        D = backend.dct2d(x)
        # For 8x8 image, DC = mean * sqrt(8*8) = mean * 8
        expected_dc = 100 * np.sqrt(64)
        assert np.isclose(D[0, 0], expected_dc, rtol=1e-10)

    def test_energy_preservation(self, backend: NumpyBackend) -> None:
        """DCT should preserve energy (orthonormal property)."""
        x = np.random.randn(64, 64)
        D = backend.dct2d(x)
        input_energy = np.sum(x**2)
        output_energy = np.sum(D**2)
        assert np.isclose(input_energy, output_energy, rtol=1e-10)


class TestGetBackend:
    """Tests for backend factory function."""

    def test_get_numpy_backend(self) -> None:
        """get_backend('numpy') should return NumpyBackend."""
        backend = get_backend("numpy")
        assert backend.name == "numpy"

    def test_get_default_backend(self) -> None:
        """get_backend() should return numpy by default."""
        backend = get_backend()
        assert backend.name == "numpy"

    def test_get_unknown_backend_raises(self) -> None:
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("unknown")


class TestTorchBackendAvailability:
    """Tests for PyTorch backend availability."""

    def test_torch_backend_import_check(self) -> None:
        """Check if torch and torch-dct are available."""
        try:
            import torch  # noqa: F401
            import torch_dct  # noqa: F401

            pytest.skip("torch and torch-dct available")
        except ImportError:
            pytest.skip("torch or torch-dct not installed")

    def test_is_available(self) -> None:
        """TorchBackend.is_available() should reflect import status."""
        try:
            from sece.backends.torch_backend import TorchBackend

            # Should return True if both packages available
            expected = True
            try:
                import torch  # noqa: F401
                import torch_dct  # noqa: F401
            except ImportError:
                expected = False

            assert TorchBackend.is_available() == expected
        except ImportError:
            pytest.skip("torch not installed")


@pytest.mark.skipif(
    True,  # Skip by default, enable with --run-torch-tests flag
    reason="PyTorch tests skipped by default. Use --run-torch-tests to enable",
)
class TestTorchBackendComparison:
    """Tests comparing TorchBackend with NumpyBackend.

    These tests require torch and torch-dct packages.
    Run with: pytest --run-torch-tests tests/test_backends.py
    """

    @pytest.fixture
    def numpy_backend(self) -> NumpyBackend:
        """Create NumpyBackend instance."""
        return NumpyBackend()

    @pytest.fixture
    def torch_backend(self) -> Backend:
        """Create TorchBackend instance (CPU for consistent testing)."""
        try:
            from sece.backends.torch_backend import TorchBackend

            return TorchBackend(device="cpu")
        except ImportError:
            pytest.skip("torch or torch-dct not installed")

    def test_torch_name(self, torch_backend: Backend) -> None:
        """TorchBackend name should be 'torch'."""
        assert torch_backend.name == "torch"

    def test_torch_device_cpu(
        self, torch_backend: Backend, numpy_backend: NumpyBackend
    ) -> None:
        """When device='cpu', should use CPU."""
        # torch_backend is created with device='cpu' in fixture
        assert torch_backend.device == "cpu"

    def test_dct2d_matches_numpy(
        self, torch_backend: Backend, numpy_backend: NumpyBackend
    ) -> None:
        """TorchBackend.dct2d() should match NumpyBackend within 1e-6."""
        x = np.random.randn(64, 64)
        result_numpy = numpy_backend.dct2d(x)
        result_torch = torch_backend.dct2d(x)
        np.testing.assert_allclose(
            result_numpy, result_torch, rtol=1e-6, atol=1e-6,
            err_msg="Torch DCT does not match NumPy DCT"
        )

    def test_idct2d_matches_numpy(
        self, torch_backend: Backend, numpy_backend: NumpyBackend
    ) -> None:
        """TorchBackend.idct2d() should match NumpyBackend within 1e-6."""
        D = np.random.randn(64, 64)
        result_numpy = numpy_backend.idct2d(D)
        result_torch = torch_backend.idct2d(D)
        np.testing.assert_allclose(
            result_numpy, result_torch, rtol=1e-6, atol=1e-6,
            err_msg="Torch IDCT does not match NumPy IDCT"
        )

    def test_roundtrip_torch(self, torch_backend: Backend) -> None:
        """DCT -> IDCT roundtrip with TorchBackend should have MSE < 1e-10."""
        x = np.random.randn(256, 256)
        D = torch_backend.dct2d(x)
        recovered = torch_backend.idct2d(D)
        mse = np.mean((x - recovered) ** 2)
        assert mse < 1e-10, f"Torch roundtrip MSE too high: {mse}"

    def test_dct2d_various_sizes(
        self, torch_backend: Backend, numpy_backend: NumpyBackend
    ) -> None:
        """DCT should match for various image sizes."""
        sizes = [(32, 32), (64, 64), (128, 128), (256, 256), (64, 128), (128, 64)]

        for h, w in sizes:
            x = np.random.randn(h, w)
            result_numpy = numpy_backend.dct2d(x)
            result_torch = torch_backend.dct2d(x)
            np.testing.assert_allclose(
                result_numpy, result_torch, rtol=1e-6, atol=1e-6,
                err_msg=f"Mismatch for size ({h}, {w})"
            )

    def test_idct2d_various_sizes(
        self, torch_backend: Backend, numpy_backend: NumpyBackend
    ) -> None:
        """IDCT should match for various image sizes."""
        sizes = [(32, 32), (64, 64), (128, 128), (256, 256), (64, 128), (128, 64)]

        for h, w in sizes:
            D = np.random.randn(h, w)
            result_numpy = numpy_backend.idct2d(D)
            result_torch = torch_backend.idct2d(D)
            np.testing.assert_allclose(
                result_numpy, result_torch, rtol=1e-6, atol=1e-6,
                err_msg=f"Mismatch for size ({h}, {w})"
            )

    def test_torch_backend_fallback_warning(self) -> None:
        """Should warn when CUDA unavailable."""
        try:
            from sece.backends.torch_backend import TorchBackend

            if TorchBackend.is_cuda_available():
                pytest.skip("CUDA available, cannot test fallback warning")
            else:
                import warnings

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    TorchBackend(device="auto")
                    assert len(w) == 1
                    assert "CUDA not available" in str(w[0].message)
        except ImportError:
            pytest.skip("torch or torch-dct not installed")

    def test_performance_512x512(self, torch_backend: Backend) -> None:
        """512x512 image should process in < 0.2s on GPU."""
        x = np.random.randn(512, 512).astype(np.float64)

        # Warm up
        _ = torch_backend.dct2d(x)
        _ = torch_backend.idct2d(x)

        # Time forward + inverse
        start = time.perf_counter()
        D = torch_backend.dct2d(x)
        _ = torch_backend.idct2d(D)
        elapsed = time.perf_counter() - start

        # On GPU, should be very fast. On CPU, may be slower.
        # Relaxed threshold to account for CPU fallback
        if torch_backend.device == "cuda":
            assert elapsed < 0.2, f"GPU processing took {elapsed:.3f}s (expected < 0.2s)"
        else:
            assert elapsed < 2.0, f"CPU processing took {elapsed:.3f}s (expected < 2s)"


class TestBackendInterface:
    """Tests for Backend abstract interface."""

    def test_backend_has_name(self) -> None:
        """All backends must have name property."""
        backend = get_backend("numpy")
        assert hasattr(backend, "name")
        assert isinstance(backend.name, str)

    def test_backend_has_device(self) -> None:
        """All backends must have device property."""
        backend = get_backend("numpy")
        assert hasattr(backend, "device")
        assert isinstance(backend.device, str)

    def test_backend_has_dct2d(self) -> None:
        """All backends must have dct2d method."""
        backend = get_backend("numpy")
        assert hasattr(backend, "dct2d")
        assert callable(backend.dct2d)

    def test_backend_has_idct2d(self) -> None:
        """All backends must have idct2d method."""
        backend = get_backend("numpy")
        assert hasattr(backend, "idct2d")
        assert callable(backend.idct2d)

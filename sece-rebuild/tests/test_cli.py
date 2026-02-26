"""Tests for SECE CLI."""

from pathlib import Path
import tempfile
import shutil

import pytest
from click.testing import CliRunner

from sece.cli.main import cli


@pytest.fixture
def runner():
    """Create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_images_dir():
    """Get path to sample images directory."""
    return Path(__file__).parent.parent / "data" / "sample_images"


class TestCLIHelp:
    """Tests for CLI help and version commands."""

    def test_help_command(self, runner):
        """Test --help shows usage information."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SECE: Spatial Entropy-based Contrast Enhancement" in result.output
        assert "--method" in result.output
        assert "--gamma" in result.output
        assert "--format" in result.output

    def test_version_command(self, runner):
        """Test --version shows version information."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "sece, version" in result.output


class TestSingleImageProcessing:
    """Tests for single image processing."""

    def test_single_image_sece(self, runner, sample_images_dir, temp_dir):
        """Test processing a single image with SECE method."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli, [str(input_path), "-o", str(output_path), "--method", "sece"]
        )

        assert result.exit_code == 0
        assert output_path.exists()

    def test_single_image_secedct(self, runner, sample_images_dir, temp_dir):
        """Test processing a single image with SECEDCT method."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli, [str(input_path), "-o", str(output_path), "--method", "secedct"]
        )

        assert result.exit_code == 0
        assert output_path.exists()

    def test_single_image_with_gamma(self, runner, sample_images_dir, temp_dir):
        """Test processing with custom gamma value."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli,
            [
                str(input_path),
                "-o",
                str(output_path),
                "--method",
                "secedct",
                "--gamma",
                "0.7",
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()

    def test_single_image_verbose(self, runner, sample_images_dir, temp_dir):
        """Test verbose output."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli, [str(input_path), "-o", str(output_path), "-v"]
        )

        assert result.exit_code == 0
        assert "Processed:" in result.output
        assert "Time:" in result.output

    def test_single_image_format_jpg(self, runner, sample_images_dir, temp_dir):
        """Test output format selection."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.jpg"

        result = runner.invoke(
            cli, [str(input_path), "-o", str(output_path), "--format", "jpg"]
        )

        assert result.exit_code == 0
        assert output_path.exists()


class TestBatchProcessing:
    """Tests for batch folder processing."""

    def test_batch_processing(self, runner, sample_images_dir, temp_dir):
        """Test processing a folder of images."""
        output_dir = temp_dir / "batch_output"

        result = runner.invoke(
            cli, [str(sample_images_dir), "-o", str(output_dir), "--method", "sece"]
        )

        assert result.exit_code == 0
        assert output_dir.exists()
        # Check that at least some images were processed
        output_images = list(output_dir.glob("*.png"))
        assert len(output_images) >= 5

    def test_batch_processing_summary(self, runner, sample_images_dir, temp_dir):
        """Test batch processing shows summary."""
        output_dir = temp_dir / "batch_output"

        result = runner.invoke(
            cli, [str(sample_images_dir), "-o", str(output_dir)]
        )

        assert result.exit_code == 0
        assert "Processing Summary" in result.output
        assert "Successful" in result.output


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_invalid_gamma_high(self, runner, sample_images_dir, temp_dir):
        """Test gamma validation rejects values > 1."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli, [str(input_path), "-o", str(output_path), "--gamma", "1.5"]
        )

        assert result.exit_code != 0
        assert "Gamma must be between 0 and 1" in result.output

    def test_invalid_gamma_negative(self, runner, sample_images_dir, temp_dir):
        """Test gamma validation rejects negative values."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli, [str(input_path), "-o", str(output_path), "--gamma", "-0.5"]
        )

        assert result.exit_code != 0
        assert "Gamma must be between 0 and 1" in result.output

    def test_invalid_format(self, runner, sample_images_dir, temp_dir):
        """Test format validation rejects unsupported formats."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.xyz"

        result = runner.invoke(
            cli, [str(input_path), "-o", str(output_path), "--format", "xyz"]
        )

        assert result.exit_code != 0
        assert "Unsupported format" in result.output

    def test_invalid_method(self, runner, sample_images_dir, temp_dir):
        """Test method validation rejects invalid methods."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli, [str(input_path), "-o", str(output_path), "--method", "invalid"]
        )

        assert result.exit_code != 0

    def test_missing_input(self, runner, temp_dir):
        """Test error when input file doesn't exist."""
        output_path = temp_dir / "output.png"

        result = runner.invoke(cli, ["nonexistent.png", "-o", str(output_path)])

        assert result.exit_code != 0

    def test_missing_output(self, runner, sample_images_dir):
        """Test error when output is not specified."""
        input_path = sample_images_dir / "camera.png"

        result = runner.invoke(cli, [str(input_path)])

        assert result.exit_code != 0
        assert "required" in result.output.lower() or "Missing" in result.output


class TestColorImageHandling:
    """Tests for color image processing."""

    def test_color_image_converts_to_grayscale(
        self, runner, sample_images_dir, temp_dir
    ):
        """Test that color images are processed (converted to grayscale)."""
        input_path = sample_images_dir / "chelsea.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli, [str(input_path), "-o", str(output_path), "-v"]
        )

        # Should succeed with warning
        assert result.exit_code == 0
        assert output_path.exists()
        assert "grayscale" in result.output.lower()


class TestMetricsPlaceholder:
    """Tests for metrics functionality (placeholder)."""

    def test_metrics_option_accepted(self, runner, sample_images_dir, temp_dir):
        """Test that metrics option is accepted."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli,
            [str(input_path), "-o", str(output_path), "--metrics", "emeg,ssim"],
        )

        # Should succeed even though metrics aren't fully implemented
        assert result.exit_code == 0

    def test_invalid_metric(self, runner, sample_images_dir, temp_dir):
        """Test that invalid metric is rejected."""
        input_path = sample_images_dir / "camera.png"
        output_path = temp_dir / "output.png"

        result = runner.invoke(
            cli,
            [str(input_path), "-o", str(output_path), "--metrics", "invalid_metric"],
        )

        assert result.exit_code != 0
        assert "Unknown metrics" in result.output

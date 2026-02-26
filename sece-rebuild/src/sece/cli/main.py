"""SECE CLI Main Module.

Command-line interface for SECE and SECEDCT image contrast enhancement.

Usage:
    sece input.png -o output.png
    sece ./folder/ -o ./output/ --method secedct --gamma 0.5
    sece image.png -o enhanced.png --format jpg --metrics emeg,ssim
"""

from pathlib import Path
from typing import Optional, List
import sys

import click
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import print as rprint

from sece.core import sece
from sece.secedct import secedct
from sece.io import load_image, save_image, ImageLoadError, UnsupportedFormatError

console = Console()

# Supported output formats
OUTPUT_FORMATS = ["png", "jpg", "jpeg", "tiff", "bmp", "webp"]

# Available metrics (placeholder - will be implemented in metrics module)
AVAILABLE_METRICS = ["emeg", "gmsd", "ssim"]


def validate_format(ctx, param, value: Optional[str]) -> Optional[str]:
    """Validate output format."""
    if value is None:
        return None
    value = value.lower()
    if value not in OUTPUT_FORMATS:
        raise click.BadParameter(
            f"Unsupported format '{value}'. Choose from: {', '.join(OUTPUT_FORMATS)}"
        )
    return value


def validate_gamma(ctx, param, value: Optional[float]) -> Optional[float]:
    """Validate gamma parameter."""
    if value is None:
        return None
    if not 0 <= value <= 1:
        raise click.BadParameter("Gamma must be between 0 and 1")
    return value


def validate_metrics(ctx, param, value: Optional[str]) -> Optional[List[str]]:
    """Validate and parse metrics list."""
    if value is None:
        return None
    metrics = [m.strip().lower() for m in value.split(",")]
    invalid = [m for m in metrics if m not in AVAILABLE_METRICS]
    if invalid:
        raise click.BadParameter(
            f"Unknown metrics: {', '.join(invalid)}. Available: {', '.join(AVAILABLE_METRICS)}"
        )
    return metrics


def get_output_path(input_path: Path, output_dir: Path, fmt: str) -> Path:
    """Generate output path for an image."""
    stem = input_path.stem
    ext = f".{fmt}" if fmt else input_path.suffix
    return output_dir / f"{stem}{ext}"


def process_single_image(
    input_path: Path,
    output_path: Path,
    method: str,
    gamma: float,
    fmt: Optional[str],
    metrics: Optional[List[str]],
    verbose: bool,
) -> dict:
    """Process a single image and return results."""
    # Load image
    image = load_image(str(input_path))

    # Handle color images by converting to grayscale for now
    # (Color support will be added in color enhancement task)
    if image.ndim == 3:
        import cv2

        original_shape = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if verbose:
            rprint(
                f"[yellow]Warning:[/yellow] Color image converted to grayscale for processing"
            )
    else:
        original_shape = image.shape

    # Apply enhancement
    if method == "sece":
        result = sece(image)
        enhanced = result.image
    else:  # secedct
        result = secedct(image, gamma=gamma)
        enhanced = result.image

    # Handle color restoration if original was color
    if len(original_shape) == 3:
        # For now, just save grayscale (color restoration in future task)
        pass

    # Save result
    save_image(enhanced, str(output_path))

    # Calculate metrics (placeholder - will be implemented)
    metric_results = {}
    if metrics:
        for metric in metrics:
            # Placeholder values
            metric_results[metric] = None

    return {
        "input": str(input_path),
        "output": str(output_path),
        "method": method,
        "gamma": gamma if method == "secedct" else None,
        "processing_time_ms": result.processing_time_ms,
        "metrics": metric_results,
    }


def collect_images(input_path: Path) -> List[Path]:
    """Collect all image files from input path."""
    supported_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

    if input_path.is_file():
        return [input_path]

    images = []
    for ext in supported_extensions:
        images.extend(input_path.rglob(f"*{ext}"))
        images.extend(input_path.rglob(f"*{ext.upper()}"))

    return sorted(set(images))


@click.command()
@click.argument(
    "input",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file or directory path",
)
@click.option(
    "-m",
    "--method",
    type=click.Choice(["sece", "secedct"], case_sensitive=False),
    default="sece",
    show_default=True,
    help="Enhancement method",
)
@click.option(
    "-g",
    "--gamma",
    type=float,
    default=0.5,
    show_default=True,
    callback=validate_gamma,
    help="Local contrast level for SECEDCT (0-1)",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=str,
    callback=validate_format,
    help="Output format (png, jpg, tiff, bmp, webp)",
)
@click.option(
    "--metrics",
    type=str,
    callback=validate_metrics,
    help="Comma-separated metrics to calculate (emeg,gmsd,ssim)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.version_option(version="0.1.0", prog_name="sece")
def cli(
    input: Path,
    output: Path,
    method: str,
    gamma: float,
    output_format: Optional[str],
    metrics: Optional[List[str]],
    verbose: bool,
):
    """SECE: Spatial Entropy-based Contrast Enhancement.

    Enhance image contrast using SECE (global) or SECEDCT (global + local) algorithms.

    \b
    Examples:
        sece input.png -o output.png
        sece ./photos/ -o ./enhanced/ --method secedct --gamma 0.7
        sece image.jpg -o enhanced.png --format png --metrics emeg,ssim
    """
    # Collect images
    images = collect_images(input)

    if not images:
        console.print("[red]Error:[/red] No supported images found", style="red")
        sys.exit(1)

    # Determine if batch processing
    is_batch = input.is_dir()

    # Setup output directory for batch
    if is_batch:
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    # Process images
    results = []

    if is_batch:
        # Batch processing with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Enhancing images", total=len(images))

            for img_path in images:
                try:
                    out_path = get_output_path(
                        img_path, output, output_format or "png"
                    )
                    result = process_single_image(
                        img_path,
                        out_path,
                        method,
                        gamma,
                        output_format,
                        metrics,
                        verbose,
                    )
                    results.append(result)
                except (ImageLoadError, UnsupportedFormatError) as e:
                    console.print(f"[red]Error processing {img_path}:[/red] {e}")
                except Exception as e:
                    console.print(f"[red]Unexpected error for {img_path}:[/red] {e}")

                progress.advance(task)
    else:
        # Single image processing
        try:
            out_path = output
            if output_format and output.suffix.lower() != f".{output_format}":
                out_path = output.with_suffix(f".{output_format}")

            result = process_single_image(
                input, out_path, method, gamma, output_format, metrics, verbose
            )
            results.append(result)

            if verbose:
                rprint(f"[green]Processed:[/green] {input} -> {out_path}")
                rprint(f"  Method: {method}")
                if method == "secedct":
                    rprint(f"  Gamma: {gamma}")
                rprint(f"  Time: {result['processing_time_ms']:.2f}ms")

        except (ImageLoadError, UnsupportedFormatError) as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            sys.exit(1)

    # Display summary
    if is_batch:
        successful = len(results)
        failed = len(images) - successful

        console.print()
        summary_table = Table(title="Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total images", str(len(images)))
        summary_table.add_row("Successful", str(successful))
        summary_table.add_row("Failed", str(failed))
        summary_table.add_row("Method", method)
        if method == "secedct":
            summary_table.add_row("Gamma", str(gamma))

        console.print(summary_table)

        if metrics and results:
            # Display metrics table (placeholder)
            metrics_table = Table(title="Metrics Summary")
            metrics_table.add_column("Image", style="cyan")
            for metric in metrics:
                metrics_table.add_column(metric.upper())

            for r in results[:10]:  # Show first 10
                row = [Path(r["input"]).name]
                for metric in metrics:
                    val = r["metrics"].get(metric)
                    row.append(f"{val:.4f}" if val else "N/A")
                metrics_table.add_row(*row)

            if len(results) > 10:
                metrics_table.add_row("...", *["..."] * len(metrics))

            console.print(metrics_table)


# Alias for direct import
enhance = cli


if __name__ == "__main__":
    cli()

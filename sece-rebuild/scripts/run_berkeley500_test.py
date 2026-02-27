#!/usr/bin/env python
"""
Run SECE and SECEDCT on Berkeley 500 dataset.
Measure EMEG and GMSD for all images.
Generate comparison statistics against baseline algorithms.
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# SECE imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sece.core import sece_simple
from sece.secedct import secedct_simple
from sece.metrics.emeg import emeg
from sece.metrics.gmsd import gmsd
from sece.metrics.ssim import ssim
from sece.io import load_image
from sece.baselines.ghe import ghe
from sece.baselines.clahe import clahe


def get_all_images(data_dir: Path) -> List[Path]:
    """Get all image paths from train/test/val folders."""
    images = []
    for split in ["train", "test", "val"]:
        split_dir = data_dir / "images" / split
        if split_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
                images.extend(split_dir.glob(ext))
    return sorted(images)


def process_image(image_path: Path, gamma: float = 0.5) -> Dict:
    """Process a single image with all methods and compute metrics."""
    # Load image
    image = load_image(str(image_path))

    # Convert to grayscale if color
    if image.ndim == 3:
        image = np.mean(image, axis=2).astype(np.uint8)

    results = {"filename": image_path.name}

    # Process with each method and time it
    methods = {}

    # SECE
    start = time.perf_counter()
    sece_result = sece_simple(image)
    results["sece_time"] = time.perf_counter() - start
    methods["sece"] = sece_result

    # SECEDCT with gamma=0.5
    start = time.perf_counter()
    secedct_result = secedct_simple(image, gamma=gamma)
    results["secedct_time"] = time.perf_counter() - start
    methods["secedct"] = secedct_result

    # GHE (baseline)
    start = time.perf_counter()
    ghe_result = ghe(image)
    results["ghe_time"] = time.perf_counter() - start
    methods["ghe"] = ghe_result

    # CLAHE (baseline)
    start = time.perf_counter()
    clahe_result = clahe(image)
    results["clahe_time"] = time.perf_counter() - start
    methods["clahe"] = clahe_result

    # Compute metrics for original
    results["emeg_original"] = emeg(image)

    # Compute metrics for each method
    for method_name, enhanced in methods.items():
        # EMEG (higher is better for contrast)
        results[f"emeg_{method_name}"] = emeg(enhanced)

        # GMSD (lower is better for distortion)
        results[f"gmsd_{method_name}"] = gmsd(image, enhanced)

        # SSIM (higher is better for similarity)
        results[f"ssim_{method_name}"] = ssim(image, enhanced)

    return results


def run_tests(data_dir: Path, output_dir: Path, gamma: float = 0.5) -> Dict:
    """Run all tests and generate reports."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    images = get_all_images(data_dir)
    print(f"Found {len(images)} images")

    if len(images) == 0:
        raise ValueError(f"No images found in {data_dir}")

    # Process all images
    all_results = []

    for image_path in tqdm(images, desc="Processing images"):
        try:
            result = process_image(image_path, gamma=gamma)
            all_results.append(result)
        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            all_results.append({
                "filename": image_path.name,
                "error": str(e)
            })

    # Save EMEG scores
    emeg_path = output_dir / "emeg_scores.csv"
    with open(emeg_path, "w", newline="") as f:
        fieldnames = ["filename", "emeg_original", "emeg_sece", "emeg_secedct", "emeg_ghe", "emeg_clahe"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            if "error" not in r:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved EMEG scores to {emeg_path}")

    # Save GMSD scores
    gmsd_path = output_dir / "gmsd_scores.csv"
    with open(gmsd_path, "w", newline="") as f:
        fieldnames = ["filename", "gmsd_sece", "gmsd_secedct", "gmsd_ghe", "gmsd_clahe"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            if "error" not in r:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved GMSD scores to {gmsd_path}")

    # Compute summary statistics
    successful_results = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]

    summary = {
        "total_images": len(images),
        "successful": len(successful_results),
        "errors": len(errors),
        "error_details": [{"filename": r["filename"], "error": r["error"]} for r in errors],
        "metrics": {}
    }

    # Compute mean and std for each metric
    metrics_to_summarize = [
        "emeg_original", "emeg_sece", "emeg_secedct", "emeg_ghe", "emeg_clahe",
        "gmsd_sece", "gmsd_secedct", "gmsd_ghe", "gmsd_clahe",
        "ssim_sece", "ssim_secedct", "ssim_ghe", "ssim_clahe",
        "sece_time", "secedct_time", "ghe_time", "clahe_time"
    ]

    for metric in metrics_to_summarize:
        values = [r[metric] for r in successful_results if metric in r]
        if values:
            summary["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }

    # Check timing constraint
    avg_sece_time = summary["metrics"].get("sece_time", {}).get("mean", 999)
    avg_secedct_time = summary["metrics"].get("secedct_time", {}).get("mean", 999)
    summary["timing_pass"] = avg_sece_time < 2.0 and avg_secedct_time < 2.0

    # Save summary
    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary statistics to {summary_path}")

    return summary


def main():
    data_dir = Path(__file__).parent.parent / "data" / "berkeley500"
    output_dir = Path(__file__).parent.parent / "results"

    print("=" * 60)
    print("Berkeley 500 Dataset Testing")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    summary = run_tests(data_dir, output_dir)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images: {summary['total_images']}")
    print(f"Successful: {summary['successful']}")
    print(f"Errors: {summary['errors']}")
    print(f"Timing pass (<2s per image): {summary['timing_pass']}")
    print()

    # Print metric comparisons
    print("EMEG Scores (higher = better contrast):")
    for method in ["original", "sece", "secedct", "ghe", "clahe"]:
        key = f"emeg_{method}"
        if key in summary["metrics"]:
            print(f"  {method}: {summary['metrics'][key]['mean']:.4f} +/- {summary['metrics'][key]['std']:.4f}")

    print()
    print("GMSD Scores (lower = less distortion):")
    for method in ["sece", "secedct", "ghe", "clahe"]:
        key = f"gmsd_{method}"
        if key in summary["metrics"]:
            print(f"  {method}: {summary['metrics'][key]['mean']:.4f} +/- {summary['metrics'][key]['std']:.4f}")

    print()
    print("Processing Times (seconds):")
    for method in ["sece", "secedct", "ghe", "clahe"]:
        key = f"{method}_time"
        if key in summary["metrics"]:
            print(f"  {method}: {summary['metrics'][key]['mean']:.4f}s +/- {summary['metrics'][key]['std']:.4f}s")


if __name__ == "__main__":
    main()

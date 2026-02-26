"""Generate comparison charts for SECE documentation."""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray

# Import SECE modules
import sys
sys.path.insert(0, 'src')

from sece.core import sece
from sece.secedct import secedct
from sece.metrics.emeg import emeg
from sece.metrics.ssim import ssim
from sece.metrics.gmsd import gmsd
from sece.baselines.ghe import ghe
from sece.baselines.clahe import clahe
from sece.baselines.wthe import wthe


def generate_comparison_charts():
    """Generate all comparison charts."""

    # Load test images
    images = {
        'Astronaut': (rgb2gray(data.astronaut()) * 255).astype(np.uint8),
        'Camera': data.camera(),
        'Coins': data.coins(),
    }

    # Create low-contrast versions
    low_contrast_images = {}
    for name, img in images.items():
        low_contrast_images[name] = np.clip(
            img.astype(np.float64) * 0.3 + 80, 0, 255
        ).astype(np.uint8)

    # Collect results
    all_results = {}

    for name in images:
        img = low_contrast_images[name]

        sece_result = sece(img)
        secedct_result = secedct(img, gamma=0.5)
        secedct_high = secedct(img, gamma=1.0)

        all_results[name] = {
            'original': img,
            'sece': sece_result.image,
            'secedct_05': secedct_result.image,
            'secedct_10': secedct_high.image,
            'ghe': ghe(img),
            'clahe': clahe(img),
            'wthe': wthe(img),
        }

    # Figure 1: EMEG Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['Original', 'SECE', 'SECEDCT(0.5)', 'SECEDCT(1.0)', 'GHE', 'CLAHE', 'WTHE']
    keys = ['original', 'sece', 'secedct_05', 'secedct_10', 'ghe', 'clahe', 'wthe']
    colors = ['#808080', '#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b']

    x = np.arange(len(images))
    width = 0.1

    for i, (method, key, color) in enumerate(zip(methods, keys, colors)):
        scores = [emeg(all_results[name][key]) for name in images]
        bars = ax.bar(x + i * width, scores, width, label=method, color=color, alpha=0.8)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_ylabel('EMEG Score', fontsize=12)
    ax.set_title('Enhancement Quality Comparison (EMEG - Higher is Better)', fontsize=14)
    ax.set_xticks(x + width * 3)
    ax.set_xticklabels(images.keys(), fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/comparison_charts.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Generated: results/comparison_charts.png")

    # Figure 2: Sample Image Enhancement Comparison
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    sample_name = 'Camera'
    sample_results = all_results[sample_name]

    display_methods = [
        ('Original', 'original'),
        ('SECE', 'sece'),
        ('SECEDCT (gamma=0.5)', 'secedct_05'),
        ('GHE', 'ghe'),
    ]

    # Row 1: Images
    for i, (label, key) in enumerate(display_methods):
        axes[0, i].imshow(sample_results[key], cmap='gray')
        axes[0, i].set_title(label)
        axes[0, i].axis('off')

    # Row 2: Histograms
    for i, (label, key) in enumerate(display_methods):
        axes[1, i].hist(sample_results[key].ravel(), bins=256, range=(0, 256),
                       color='steelblue', alpha=0.7)
        axes[1, i].set_title(f'{label} Histogram')
        axes[1, i].set_xlim(0, 256)

    plt.suptitle('Sample Enhancement Comparison (Camera Image)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/sample_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Generated: results/sample_comparison.png")

    # Figure 3: Gamma Parameter Effect
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample_img = low_contrast_images['Camera']

    for i, gamma in enumerate(gammas):
        result = secedct(sample_img, gamma=gamma)
        axes[0, i].imshow(result.image, cmap='gray')
        axes[0, i].set_title(f'gamma={gamma}\nalpha={result.alpha:.2f}')
        axes[0, i].axis('off')

    # Row 2: Local contrast detail (center crop)
    crop = slice(100, 200), slice(150, 250)
    for i, gamma in enumerate(gammas):
        result = secedct(sample_img, gamma=gamma)
        axes[1, i].imshow(result.image[crop], cmap='gray')
        axes[1, i].set_title('Detail')
        axes[1, i].axis('off')

    plt.suptitle('Effect of Gamma Parameter on Local Contrast', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/gamma_effect.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Generated: results/gamma_effect.png")

    # Figure 4: Quality vs Distortion Trade-off
    fig, ax = plt.subplots(figsize=(10, 8))

    for method, key, color, marker in [
        ('SECE', 'sece', '#1f77b4', 'o'),
        ('SECEDCT(0.5)', 'secedct_05', '#2ca02c', 's'),
        ('SECEDCT(1.0)', 'secedct_10', '#d62728', '^'),
        ('GHE', 'ghe', '#ff7f0e', 'D'),
        ('CLAHE', 'clahe', '#9467bd', 'v'),
        ('WTHE', 'wthe', '#8c564b', 'p'),
    ]:
        for name in images:
            original = low_contrast_images[name]
            enhanced = all_results[name][key]
            q_score = emeg(enhanced)
            d_score = gmsd(original, enhanced)
            ax.scatter(d_score, q_score, c=color, marker=marker, s=100,
                      label=method if name == 'Camera' else '', alpha=0.7)

    ax.set_xlabel('Distortion (GMSD) - Lower is Better', fontsize=12)
    ax.set_ylabel('Quality (EMEG) - Higher is Better', fontsize=12)
    ax.set_title('Quality vs Distortion Trade-off', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    # Add annotations for ideal region
    ax.annotate('Ideal: High Quality, Low Distortion',
                xy=(0.02, 0.6), fontsize=10, color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.savefig('results/quality_distortion_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Generated: results/quality_distortion_tradeoff.png")

    print("\nAll charts generated successfully!")


if __name__ == '__main__':
    generate_comparison_charts()

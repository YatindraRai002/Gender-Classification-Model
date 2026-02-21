"""
analyze_data.py — Exploratory Data Analysis (EDA) for the gender classification dataset.

Run:
    python analyze_data.py

Outputs:
    - Console: class distribution table, image size statistics, corruption report
    - outputs/class_distribution.png
    - outputs/image_size_distribution.png
"""

import os
import sys
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Dataset")
SPLITS = ["Train", "Validation", "Test"]
CLASSES = ["Male", "Female"]
OUTPUT_DIR = "outputs"


def count_images(data_dir: str) -> dict:
    """
    Count images per class per split.

    Returns:
        Nested dict: {split: {class: count}}
    """
    counts = {}
    for split in SPLITS:
        counts[split] = {}
        for cls in CLASSES:
            folder = os.path.join(data_dir, split, cls)
            if os.path.exists(folder):
                n = len([f for f in os.listdir(folder)
                         if os.path.isfile(os.path.join(folder, f))])
                counts[split][cls] = n
            else:
                counts[split][cls] = 0
                print(f"[WARNING] Folder not found: {folder}")
    return counts


def print_distribution(counts: dict):
    """Print a formatted distribution table."""
    print("\n" + "=" * 65)
    print("DATASET DISTRIBUTION")
    print("=" * 65)
    print(f"{'Split':<15} {'Male':>10} {'Female':>10} {'Total':>10} {'Male %':>10}")
    print("-" * 65)
    for split in SPLITS:
        male = counts[split]["Male"]
        female = counts[split]["Female"]
        total = male + female
        male_pct = male / total * 100 if total > 0 else 0
        print(f"{split:<15} {male:>10,} {female:>10,} {total:>10,} {male_pct:>9.1f}%")
    print("=" * 65)

    # Overall
    total_male = sum(counts[s]["Male"] for s in SPLITS)
    total_female = sum(counts[s]["Female"] for s in SPLITS)
    total = total_male + total_female
    print(f"{'TOTAL':<15} {total_male:>10,} {total_female:>10,} {total:>10,} {total_male / total * 100:>9.1f}%")

    # Imbalance analysis
    print("\n📊 IMBALANCE ANALYSIS:")
    ratio = total_female / total_male if total_male > 0 else 0
    print(f"  Female:Male ratio = {ratio:.2f}:1")
    if ratio > 1.3 or ratio < 0.77:
        print("  ⚠️  Moderate class imbalance detected!")
        print("  → Recommendation: Use class weights during training")
        print(f"  → Suggested weights: Male={total / (2 * total_male):.4f}, "
              f"Female={total / (2 * total_female):.4f}")
    else:
        print("  ✅ Classes are reasonably balanced")


def plot_distribution(counts: dict, save_dir: str):
    """Plot class distribution bar charts."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"Male": "#3498db", "Female": "#e74c3c"}

    for idx, split in enumerate(SPLITS):
        male = counts[split]["Male"]
        female = counts[split]["Female"]

        bars = axes[idx].bar(
            ["Male", "Female"],
            [male, female],
            color=[colors["Male"], colors["Female"]],
            edgecolor="white",
            linewidth=1.5,
        )
        axes[idx].set_title(f"{split} Set", fontsize=14, fontweight="bold")
        axes[idx].set_ylabel("Number of Images")

        # Add count labels on bars
        for bar, val in zip(bars, [male, female]):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 100,
                f"{val:,}",
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

        axes[idx].grid(axis="y", alpha=0.3)

    plt.suptitle("Class Distribution Across Splits", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Distribution plot saved: {save_path}")
    plt.close()


def analyze_image_sizes(data_dir: str, sample_size: int = 2000) -> dict:
    """
    Sample images and analyze their dimensions.

    Args:
        data_dir: Root dataset directory.
        sample_size: Number of images to sample per split.

    Returns:
        dict with width/height statistics.
    """
    print("\n" + "=" * 65)
    print("IMAGE SIZE ANALYSIS")
    print("=" * 65)

    all_widths = []
    all_heights = []

    for split in SPLITS:
        widths, heights = [], []
        for cls in CLASSES:
            folder = os.path.join(data_dir, split, cls)
            if not os.path.exists(folder):
                continue
            files = os.listdir(folder)
            sample = files[:min(sample_size // 2, len(files))]

            for fname in sample:
                fpath = os.path.join(folder, fname)
                try:
                    with Image.open(fpath) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                except Exception:
                    pass

        all_widths.extend(widths)
        all_heights.extend(heights)

        if widths:
            print(f"\n[{split}] Sampled {len(widths)} images:")
            print(f"  Width  — min: {min(widths)}, max: {max(widths)}, "
                  f"mean: {np.mean(widths):.0f}, median: {np.median(widths):.0f}")
            print(f"  Height — min: {min(heights)}, max: {max(heights)}, "
                  f"mean: {np.mean(heights):.0f}, median: {np.median(heights):.0f}")

    # Plot
    if all_widths:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(all_widths, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
        axes[0].set_title("Image Width Distribution")
        axes[0].set_xlabel("Width (px)")
        axes[0].set_ylabel("Count")
        axes[0].axvline(224, color="red", linestyle="--", label="Target: 224px")
        axes[0].legend()

        axes[1].hist(all_heights, bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
        axes[1].set_title("Image Height Distribution")
        axes[1].set_xlabel("Height (px)")
        axes[1].set_ylabel("Count")
        axes[1].axvline(224, color="red", linestyle="--", label="Target: 224px")
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "image_size_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[INFO] Size distribution plot saved: {save_path}")
        plt.close()

    print(f"\n💡 RECOMMENDATION: Resize all images to 224×224")
    print(f"   This is the standard input size for ResNet50, EfficientNetB0, MobileNetV2")

    return {"widths": all_widths, "heights": all_heights}


def detect_corrupted_images(data_dir: str) -> list:
    """
    Scan all images and detect corrupted/unreadable files.

    Returns:
        List of corrupted file paths.
    """
    print("\n" + "=" * 65)
    print("CORRUPTION CHECK")
    print("=" * 65)

    corrupted = []
    total_checked = 0

    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(data_dir, split, cls)
            if not os.path.exists(folder):
                continue
            files = os.listdir(folder)
            for fname in tqdm(files, desc=f"Checking {split}/{cls}", leave=False):
                total_checked += 1
                fpath = os.path.join(folder, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()  # Verify image integrity
                except Exception as e:
                    corrupted.append((fpath, str(e)))

    if corrupted:
        print(f"\n⚠️  Found {len(corrupted)} corrupted images out of {total_checked:,} checked:")
        for path, error in corrupted[:20]:  # Show first 20
            print(f"  ❌ {path}: {error}")
        if len(corrupted) > 20:
            print(f"  ... and {len(corrupted) - 20} more")
        print("\n  → Recommendation: Remove corrupted images before training")
    else:
        print(f"\n✅ All {total_checked:,} images are valid!")

    return corrupted


# ============================================================
# MAIN
# ============================================================

def main():
    """Run complete data analysis pipeline."""
    print("🔍 GENDER CLASSIFICATION — DATA ANALYSIS")
    print("=" * 65)

    # 1. Count images
    counts = count_images(DATA_DIR)
    print_distribution(counts)

    # 2. Plot distributions
    plot_distribution(counts, OUTPUT_DIR)

    # 3. Analyze image sizes
    analyze_image_sizes(DATA_DIR, sample_size=2000)

    # 4. Detect corrupted images
    detect_corrupted_images(DATA_DIR)

    print("\n" + "=" * 65)
    print("✅ DATA ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()

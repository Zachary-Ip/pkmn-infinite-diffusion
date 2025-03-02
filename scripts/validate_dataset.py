import os
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt


def validate_image_dataset(
    image_folder, sample_size=None, report_file="dataset_validation_report.txt"
):
    """
    Validate a dataset of images for consistency and quality.

    Args:
        image_folder: Path to folder containing images
        sample_size: If set, only check this many random images
        report_file: Path to save the validation report
    """
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob("*.png"))

    if not image_files:
        print(f"No PNG images found in {image_folder}")
        return

    print(f"Found {len(image_files)} PNG images in {image_folder}")

    # Sample if requested
    if sample_size and sample_size < len(image_files):
        import random

        image_files = random.sample(image_files, sample_size)
        print(f"Randomly sampling {sample_size} images for validation")

    # Initialize counters and storage
    dimensions = Counter()
    modes = Counter()
    has_transparency = 0
    corrupted_files = []
    background_colors = Counter()
    sample_backgrounds = []

    # Process each image
    for img_path in tqdm(image_files, desc="Validating images"):
        try:
            with Image.open(img_path) as img:
                # Check image mode
                modes[img.mode] += 1

                # Check dimensions
                dimensions[img.size] += 1

                # Check for transparency
                if "A" in img.mode:
                    has_transparency += 1

                # Check background color (assuming corners are background)
                img_array = np.array(img)
                if img.mode == "RGBA":
                    # If image has transparency, check if any fully transparent pixels exist
                    if np.any(img_array[:, :, 3] == 0):
                        background_colors["transparent"] += 1
                    # Convert to RGB for corner color analysis
                    img_rgb = Image.new("RGB", img.size, (255, 255, 255))
                    img_rgb.paste(img, mask=img.split()[3])
                    img_array = np.array(img_rgb)

                # Get corner pixels to estimate background color
                h, w = img_array.shape[:2]
                corners = [
                    tuple(img_array[0, 0][:3]),
                    tuple(img_array[0, w - 1][:3]),
                    tuple(img_array[h - 1, 0][:3]),
                    tuple(img_array[h - 1, w - 1][:3]),
                ]

                # If all corners are the same color, consider it the background
                if len(set(corners)) == 1:
                    bg_color = corners[0]
                    background_colors[bg_color] += 1

                    # Store a few examples of different backgrounds for visualization
                    if len(sample_backgrounds) < 10 or bg_color not in [
                        b[0] for b in sample_backgrounds
                    ]:
                        sample_backgrounds.append((bg_color, img_path))
                else:
                    background_colors["inconsistent"] += 1

        except Exception as e:
            corrupted_files.append((img_path, str(e)))

    # Generate report
    with open(report_file, "w") as f:
        f.write(f"Dataset Validation Report for {image_folder}\n")
        f.write(f"Total images checked: {len(image_files)}\n\n")

        f.write("IMAGE DIMENSIONS:\n")
        for dim, count in dimensions.most_common():
            f.write(f"  {dim}: {count} images ({count/len(image_files)*100:.1f}%)\n")

        f.write("\nIMAGE MODES:\n")
        for mode, count in modes.most_common():
            f.write(f"  {mode}: {count} images ({count/len(image_files)*100:.1f}%)\n")

        f.write(f"\nTRANSPARENCY:\n")
        f.write(
            f"  Images with alpha channel: {has_transparency} ({has_transparency/len(image_files)*100:.1f}%)\n"
        )

        f.write("\nBACKGROUND COLORS:\n")
        for bg, count in background_colors.most_common():
            if bg == "transparent":
                f.write(
                    f"  Transparent: {count} images ({count/len(image_files)*100:.1f}%)\n"
                )
            elif bg == "inconsistent":
                f.write(
                    f"  Inconsistent corners: {count} images ({count/len(image_files)*100:.1f}%)\n"
                )
            else:
                f.write(
                    f"  RGB{bg}: {count} images ({count/len(image_files)*100:.1f}%)\n"
                )

        if corrupted_files:
            f.write(f"\nCORRUPTED FILES ({len(corrupted_files)}):\n")
            for path, error in corrupted_files[:20]:  # List first 20 only
                f.write(f"  {path}: {error}\n")
            if len(corrupted_files) > 20:
                f.write(f"  ... and {len(corrupted_files) - 20} more\n")

    # Print summary to console
    print(f"\nValidation complete! Report saved to {report_file}")
    print(f"Images checked: {len(image_files)}")
    print(f"Unique dimensions: {len(dimensions)}")
    print(f"Unique modes: {len(modes)}")
    print(f"Corrupted files: {len(corrupted_files)}")

    # Create a visualization of the most common background colors
    visualize_backgrounds(
        background_colors, sample_backgrounds, "background_samples.png"
    )

    return background_colors, dimensions, modes


def visualize_backgrounds(background_colors, sample_backgrounds, output_file):
    """Create a visualization of background color samples"""
    # Filter out special entries
    bg_colors = [
        (bg, count)
        for bg, count in background_colors.most_common()
        if bg not in ("transparent", "inconsistent")
    ][:6]

    if not bg_colors:
        return

    # Find matching samples for the top colors
    samples_to_show = []
    for color, _ in bg_colors:
        for sample_color, sample_path in sample_backgrounds:
            if sample_color == color and sample_path not in [
                s[1] for s in samples_to_show
            ]:
                samples_to_show.append((color, sample_path))
                break

    if not samples_to_show:
        return

    # Create the visualization
    fig, axes = plt.subplots(
        len(samples_to_show), 2, figsize=(10, 3 * len(samples_to_show))
    )
    if len(samples_to_show) == 1:
        axes = [axes]

    for i, (color, path) in enumerate(samples_to_show):
        # Color swatch
        axes[i][0].add_patch(
            plt.Rectangle((0, 0), 1, 1, color=[c / 255 for c in color])
        )
        axes[i][0].set_title(f"RGB{color}")
        axes[i][0].set_xlim(0, 1)
        axes[i][0].set_ylim(0, 1)
        axes[i][0].axis("off")

        # Sample image
        img = Image.open(path)
        axes[i][1].imshow(np.array(img.convert("RGB")))
        axes[i][1].set_title(f"Sample: {path.name}")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Background color visualization saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate image dataset for ML training"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="Path to folder containing images",
        default="data/processed/",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of random images to check (default: all)",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="dataset_validation_report.txt",
        help="Path to save report",
    )

    args = parser.parse_args()

    validate_image_dataset(args.image_folder, args.sample_size, args.report_file)

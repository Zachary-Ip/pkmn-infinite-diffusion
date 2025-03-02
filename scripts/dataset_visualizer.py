import os
import configparser
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
from src.utils.dataset import PokemonDataset


def visualize_dataset_transforms():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize dataset transformations")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset images",
        default="data/processed/",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        help="Path to metadata file",
        default="data/metadata/",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.ini",
        help="Path to config file",
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for dataloader"
    )
    args = parser.parse_args()

    # Get resolution from config.ini
    config = configparser.ConfigParser()
    config.read(args.config_path)
    resolution = int(config["settings"].get("resolution", 128))

    # Define transformations
    # Original images (minimal transform)
    orig_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # After resize
    resize_transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ]
    )

    # Full transform pipeline (same as training)
    full_transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Create datasets with different transforms
    orig_dataset = PokemonDataset(
        args.dataset_path, args.metadata_path, transform=orig_transform
    )
    resize_dataset = PokemonDataset(
        args.dataset_path, args.metadata_path, transform=resize_transform
    )
    full_dataset = PokemonDataset(
        args.dataset_path, args.metadata_path, transform=full_transform
    )

    # Create dataloaders
    orig_loader = DataLoader(orig_dataset, batch_size=args.batch_size, shuffle=True)
    resize_loader = DataLoader(resize_dataset, batch_size=args.batch_size, shuffle=True)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

    # Get the same batch from each loader (using seed for reproducibility)
    torch.manual_seed(42)
    orig_batch = next(iter(orig_loader))

    torch.manual_seed(42)
    resize_batch = next(iter(resize_loader))

    torch.manual_seed(42)
    full_batch = next(iter(full_loader))

    # Extract images from the dictionary return format
    orig_images = orig_batch["image"]
    resize_images = resize_batch["image"]
    full_images = full_batch["image"]

    # Get metadata for display
    metadata = orig_batch["metadata"]

    # Visualize samples
    num_samples = min(args.batch_size, args.num_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))

    # Handle the case when there's only one sample (axes won't be 2D)
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original image
        orig_img = orig_images[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(orig_img, 0, 1))
        axes[i, 0].set_title(f"Original")

        # Resized image
        resize_img = resize_images[i].permute(1, 2, 0).numpy()
        axes[i, 1].imshow(np.clip(resize_img, 0, 1))
        axes[i, 1].set_title(f"Resized ({resolution}x{resolution})")

        # Normalized image (need to denormalize for visualization)
        norm_img = full_images[i].permute(1, 2, 0).numpy()
        # Denormalize: pixel = (pixel * std) + mean
        denorm_img = (norm_img * 0.5) + 0.5
        axes[i, 2].imshow(np.clip(denorm_img, 0, 1))
        axes[i, 2].set_title(f"Normalized")

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs("visualization", exist_ok=True)
    plt.savefig("visualization/transform_visualization.png", dpi=300)

    # Create a second figure for metadata visualization
    if metadata is not None:
        plt.figure(figsize=(10, 6))
        for i in range(min(num_samples, len(metadata))):
            plt.subplot(num_samples, 1, i + 1)
            plt.bar(range(len(metadata[i])), metadata[i].numpy())
            plt.title(f"Sample {i+1} Metadata (One-hot encoded)")
            plt.ylabel("Value")
            plt.ylim(0, 1.1)
            # Only show x labels for the bottom plot
            if i == num_samples - 1:
                plt.xlabel("Type Index")

        plt.tight_layout()
        plt.savefig("visualization/metadata_visualization.png", dpi=300)
        print(
            f"Metadata visualization saved to visualization/metadata_visualization.png"
        )

    plt.show()
    print(
        f"Image transform visualization saved to visualization/transform_visualization.png"
    )


if __name__ == "__main__":
    visualize_dataset_transforms()

import argparse
import torch
from pathlib import Path
from src.models.unet import UNet  # Adjust import based on your repo structure
from src.utils.utils import load_model, save_images  # Ensure these functions exist
from src.training.ddim import DDIMScheduler  # Adjust import as needed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Pokémon images with specified metadata."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--types",
        type=str,
        default="",
        help="Comma-separated Pokémon types (e.g., 'fire,flying').",
    )
    parser.add_argument(
        "--egg_group",
        type=str,
        default="",
        help="Comma-separated egg groups (e.g., 'monster').",
    )
    parser.add_argument(
        "--num_images", type=int, default=1, help="Number of images to generate."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_images",
        help="Directory to save generated images.",
    )
    return parser.parse_args()


def encode_metadata(types, egg_groups, type_to_idx, egg_group_to_idx):
    """Encodes metadata into one-hot vectors based on the type and egg group indices."""
    type_vec = torch.zeros(len(type_to_idx))
    egg_vec = torch.zeros(len(egg_group_to_idx))

    for t in types.split(","):
        if t in type_to_idx:
            type_vec[type_to_idx[t]] = 1
    for e in egg_groups.split(","):
        if e in egg_group_to_idx:
            egg_vec[egg_group_to_idx[e]] = 1

    return torch.cat((type_vec, egg_vec)).unsqueeze(
        0
    )  # Shape (1, num_metadata_features)


def main():
    args = parse_args()

    # Load model
    model = UNet()  # Adjust based on your model structure
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu")["ema_model_state"]
    )
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    # Load noise scheduler
    noise_scheduler = DDIMScheduler()

    # Encode metadata
    type_to_idx = {"fire": 0, "flying": 1}  # Load actual mappings
    egg_group_to_idx = {"monster": 0}  # Load actual mappings
    metadata = encode_metadata(
        args.types, args.egg_group, type_to_idx, egg_group_to_idx
    ).to(model.device)

    # Generate images
    with torch.no_grad():
        generator = torch.manual_seed(0)
        generated_images = noise_scheduler.generate(
            model,
            num_inference_steps=50,
            generator=generator,
            eta=1.0,
            batch_size=args.num_images,
            guidance_scale=args.guidance_scale,
            metadata=metadata,
        )

    # Save images
    save_images(generated_images, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

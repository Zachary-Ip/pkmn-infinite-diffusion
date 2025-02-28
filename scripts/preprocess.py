import argparse
import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def process_directory(args):
    successful_count = 0
    failed_count = 0

    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create if not exists

    for filename in tqdm(os.listdir(args.input_directory)):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(args.input_directory, filename)
            result = fill_transparent_with_white(
                input_path,
                args.output_directory,
                delete_corrupted=args.delete_corrupted,
            )

            if result:
                successful_count += 1
            else:
                failed_count += 1

    print(
        f"Processing complete. Successfully processed: {successful_count}, Failed: {failed_count}"
    )


def fill_transparent_with_white(input_path, output_dir, delete_corrupted):
    """
    Load a PNG image with transparency, fill transparent pixels with white,
    remove the alpha channel, and save the modified image.

    Args:
        input_path (str): Path to the input PNG image with transparency
        output_path (str, optional): Path to save the output image. If None,
                                     will save as "{original_name}_filled.png"
        delete_corrupted (bool): Whether to delete corrupted files

    Returns:
        str: Path to the saved output image or None if processing failed
    """
    try:
        # Load and check image integrity
        img = _load_image(input_path, delete=delete_corrupted)

        # If image is corrupted, return None
        if img is None:
            return None

        # Check if the image has an alpha channel
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            # Create a new white background image with the same size
            background = Image.new("RGB", img.size, (255, 255, 255))

            # If image is in palette mode (P) with transparency
            if img.mode == "P" and "transparency" in img.info:
                # Convert to RGBA first to handle transparency properly
                img = img.convert("RGBA")

            # Paste the image onto the white background using alpha as mask
            background.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)

            # Save the image without alpha channel
            output_path = Path(output_dir) / Path(input_path).name
            background.save(output_path)
            return output_path
        else:
            return input_path
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return None


def _load_image(path: str, delete: bool = False):
    """
    Attempt to load an image and verify its integrity.
    If the image is corrupted, log a warning and optionally delete it.

    Args:
        path (str): Path to the image file.
        delete (bool): Flag to delete detected corrupted files.

    Returns:
        Image object if successfully loaded, None if corrupted.
    """
    try:
        # First try to verify the image
        with Image.open(path) as img:
            img.verify()

        # If verification passes, reopen and return the image
        # We need to reopen because verify() closes the file
        return Image.open(path)
    except (IOError, UnidentifiedImageError, SyntaxError, ValueError) as e:
        print(f"Corrupted file detected: {path}")
        print(f"Error: {str(e)}")

        if delete:
            try:
                os.remove(path)
                print(f"Deleted corrupted file: {path}")
            except Exception as del_err:
                print(f"Failed to delete corrupted file: {path}, error: {str(del_err)}")

        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in a directory.")
    parser.add_argument(
        "--input_directory",
        type=str,
        help="Path to the directory containing images",
        default="data/raw/",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to the directory to save images",
        default="data/processed/",
    )
    parser.add_argument(
        "--delete_corrupted",
        type=bool,
        help="Path to the directory to save images",
        default=True,
    )
    args = parser.parse_args()

    process_directory(args)

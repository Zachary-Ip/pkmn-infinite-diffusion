import os
from PIL import Image, UnidentifiedImageError
import argparse
from tqdm import tqdm
from pathlib import Path

def preprocess(directory:str):
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory '{dir_path}' does not exist.")
        return
    for file in tqdm(dir_path.glob("*")):
        if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            im = _load_image(file)
            if im:
                im = _add_background(im)
                im.convert('RGB').save(file, "PNG")

def _load_image(path:str, delete: bool = True):
    """
    Attempt to load an image and verify its integrity.
    If the image is corrupted, log a warning and return None.

    Args:
        path (str): Path to the image file.
        delete Boolean): Flag to delete detected corrupted files.

    Returns:
        Image object if successfully loaded, None if corrupted.
    """
    try:
        with Image.open(path) as im:
            im.verify()
        return Image.open(path)
    except (IOError, UnidentifiedImageError):
        print(f"Deleting corrupted file: {path}")
        if delete:
            os.remove(path)
        return None
    


def _add_background(im):
    """
    Add a white background to images with an alpha channel.

    Args:
        im (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Image with transparency removed (RGB mode).
    """
    if im.mode == "RGBA":
        
        new_im = Image.new("RGBA", im.size, "WHITE")
        new_im.paste(im,(0,0), im)
        return new_im
    else:
        print("Loaded image does not have transparency")
        return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing images")
    args = parser.parse_args()

    preprocess(args.directory)



import os

import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Define Transformations (Resize + Normalize)


config = OmegaConf.load("configs/train.yaml")

transform = transforms.Compose(
    [
        transforms.Resize(
            (config.model.image_size, config.model.image_size)
        ),  # Resize images to 64x64
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1,1]
    ]
)


class PokemonDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = self.get_valid_images(image_folder)
        self.transform = transform

    def get_valid_images(self, folder):
        valid_images = []
        for f in os.listdir(folder):
            if f.endswith(".png"):
                img_path = os.path.join(folder, f)
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify if the image is corrupted
                    valid_images.append(f)
                except (IOError, UnidentifiedImageError):
                    print(f"Skipping corrupted file: {f}")
        return valid_images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

import csv
import json
import os
import re

import torch
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
    def __init__(self, image_folder, metadata_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = self.get_valid_images(image_folder)
        self.transform = transform

        # Get the metadata
        self.metadata = self.read_data(metadata_folder, "metadata.json")
        self.types = self.read_data(metadata_folder, "types.csv")
        self.egg_groups = self.read_data(metadata_folder, "egg_groups.csv")
        self.colors = self.read_data(metadata_folder, "colors.csv")
        self.shapes = self.read_data(metadata_folder, "shapes.csv")

        # Map metadata to one-hot encoding
        self.type_to_idx = {t: i for i, t in enumerate(self.types)}
        self.egg_group_to_idx = {e: i for i, e in enumerate(self.egg_groups)}
        self.color_to_idx = {t: i for i, t in enumerate(self.colors)}
        self.shapes_group_to_idx = {e: i for i, e in enumerate(self.shapes)}

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

    @staticmethod
    def read_data(dir, file):
        path = os.path.join(dir, file)
        _, ext = os.path.splitext(file)
        with open(path, "r") as f:
            if ext == "json":
                data = json.load(f)
            elif ext == "csv":
                reader = csv.reader(f)
                data = {row[0] for row in reader}
            else:
                raise ValueError(f"File extension for {file} recognized")
        return data

    def encode_one_hot(self, labels, label_to_idx, num_classes):
        """Convert a list of labels into a one-hot encoded tensor"""
        one_hot = torch.zeros(num_classes)

    @staticmethod
    def get_ID_from_name(file):
        name, _ = os.path.splitext(file)
        idx_list = name.split(".")

        # drop any letters from the name
        return [re.sub("\D", "", s) for s in idx_list]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        pkmn_ids = self.get_ID_from_name(img_path)

        if self.transform:
            image = self.transform(image)
        return {"image": image}

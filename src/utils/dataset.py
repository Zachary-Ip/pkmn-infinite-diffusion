import csv
import json
import os
import re
from pathlib import Path

import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Load Config
config = OmegaConf.load("configs/train.yaml")

# Define Transformations (Resize + Normalize)
transform = transforms.Compose(
    [
        transforms.Resize((config.model.image_size, config.model.image_size)),  # Resize
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1,1]
    ]
)


class PokemonDataset(Dataset):
    def __init__(self, image_folder, metadata_folder, transform=None):
        self.image_folder = Path(image_folder)
        self.image_files = self.get_valid_images()
        self.transform = transform

        # Load metadata
        self.metadata = self.read_data(metadata_folder, "metadata.json")
        self.types = self.read_data(metadata_folder, "types.csv")
        # self.egg_groups = self.read_data(metadata_folder, "egg_groups.csv")
        # self.colors = self.read_data(metadata_folder, "colors.csv")
        # self.shapes = self.read_data(metadata_folder, "shapes.csv")

        # Create mappings for one-hot encoding
        self.type_to_idx = {t: i for i, t in enumerate(self.types)}
        # self.egg_group_to_idx = {e: i for i, e in enumerate(self.egg_groups)}
        # self.color_to_idx = {c: i for i, c in enumerate(self.colors)}
        # self.shape_to_idx = {s: i for i, s in enumerate(self.shapes)}

    def get_valid_images(self):
        return [
            f.name
            for f in self.image_folder.iterdir()
            if f.suffix == ".png" and self.is_valid_image(f)
        ]

    @staticmethod
    def is_valid_image(filepath):
        try:
            with Image.open(filepath) as img:
                img.verify()
            return True
        except (IOError, UnidentifiedImageError):
            print(f"Skipping corrupted file: {filepath}")
            return False

    @staticmethod
    def read_data(folder, filename):
        path = Path(folder) / filename
        ext = path.suffix

        with path.open("r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext == ".csv":
                reader = csv.reader(f)
                return next(reader)  # csv is a single row
            else:
                raise ValueError(f"Unrecognized file extension for {filename}")

    @staticmethod
    def get_ID_from_name(filename):
        return [re.sub(r"\D", "", part) for part in Path(filename).stem.split(".")]

    def encode_one_hot(self, labels, label_to_idx, num_classes):
        """Convert a list of labels into a one-hot encoded tensor."""
        one_hot = torch.zeros(num_classes)
        for label in labels:
            if label in label_to_idx:
                one_hot[label_to_idx[label]] = 1
        return one_hot

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.image_folder / img_name
        image = Image.open(img_path).convert("RGB")

        # Extract Pokémon IDs
        pkmn_ids = self.get_ID_from_name(img_name)

        # Initialize metadata sets
        type_set, egg_group_set, color_set, shape_set = set(), set(), set(), set()

        for pkmn_id in pkmn_ids:
            id_data = self.metadata.get(pkmn_id, {})
            type_set.update(id_data.get("types", []))
            egg_group_set.update(id_data.get("egg_groups", []))

        if self.transform:
            image = self.transform(image)

        metadata = self.encode_one_hot(
            type_set, self.type_to_idx, len(self.types)
        )  # torch.cat(
        #     (
        #         self.encode_one_hot(type_set, self.type_to_idx, len(self.types)),
        #         self.encode_one_hot(
        #             egg_group_set, self.egg_group_to_idx, len(self.egg_groups)
        #         ),
        #     ),
        #     0,
        # )
        # Classifier-Free Guidance probability
        drop_prob = 0.1  # 10% of the time, we drop metadata
        use_conditioning = torch.rand(1) > drop_prob

        metadata = metadata * use_conditioning  # Set to zero vector if dropped

        return {
            "image": image,
            "metadata": metadata,
        }

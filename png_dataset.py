# png_dataset.py
import glob
import os
from typing import List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PNGDataset(Dataset):
    def __init__(self, root_dir: str):
        """
        root_dir:
            Directory containing PNG images (can have subfolders).
        image_size:
            Target resolution (Stable Diffusion VAE typically uses 512).
        """
        self.root_dir = root_dir
        # Recursively find PNG files
        self.image_paths: List[str] = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG files found under {root_dir}")

        # Transform: PIL -> tensor in [-1, 1]
        self.transform = T.Compose(
            [
                T.ConvertImageDtype(torch.float32) if hasattr(T, "ConvertImageDtype") else T.Lambda(lambda x: x),
                T.ToTensor(),
                # [0, 1] -> [-1, 1]
                T.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # Match the HF dataset format your code expects
        return {"pixel_values": img}

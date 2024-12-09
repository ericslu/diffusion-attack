import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

def add_gaussian_noise(image, mean=0, std=0.1):
    """
    Adds Gaussian noise to the image.
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

class MixedDataset(Dataset):
    def __init__(self, root_dir, noise_fraction=0.5, noise_type="gaussian", transform=None):
        """
        Args:
            root_dir: Root directory containing subdirectories 'Real' and 'Fake'.
            noise_fraction: Fraction of images to make noisy (e.g., 0.5 means 50% noisy).
            noise_type: Type of noise to add ('gaussian' supported here).
            transform: Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.noise_fraction = noise_fraction
        self.noise_type = noise_type
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, subdir in enumerate(["Real", "Fake"]):
            subdir_path = os.path.join(self.root_dir, subdir)
            for file_name in os.listdir(subdir_path):
                if file_name.endswith(".png") or file_name.endswith(".jpg"):
                    self.image_paths.append(os.path.join(subdir_path, file_name))
                    self.labels.append(label)

        self.num_images = len(self.image_paths)
        self.noisy_indices = set(np.random.choice(self.num_images, int(self.noise_fraction * self.num_images), replace=False))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        image = np.array(image) / 255.0

        if idx in self.noisy_indices:
            if self.noise_type == "gaussian":
                image = add_gaussian_noise(image)

        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

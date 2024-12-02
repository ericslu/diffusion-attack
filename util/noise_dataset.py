import numpy as np
import torch
from torch.utils.data import Dataset

def add_gaussian_noise(image, mean=0, std=0.1):
    """
    Adds Gaussian noise to the image.
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)  # Ensure pixel values are in range [0, 1]


class MixedDataset(Dataset):
    def __init__(self, image_dataset, noise_fraction=0.5, noise_type="gaussian", transform=None):
        """
        Args:
            image_dataset: Original dataset (e.g., CIFAR-10, custom dataset).
            noise_fraction: Fraction of images to make noisy (e.g., 0.5 means 50% noisy).
            noise_type: Type of noise to add ('gaussian' supported here).
            transform: Transformations to apply to the images.
        """
        self.image_dataset = image_dataset
        self.noise_fraction = noise_fraction
        self.noise_type = noise_type
        self.transform = transform

        # Randomly select indices for noisy images
        self.num_images = len(self.image_dataset)
        self.noisy_indices = set(np.random.choice(self.num_images, int(self.noise_fraction * self.num_images), replace=False))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]

        # Check if the current index is in the noisy subset
        if idx in self.noisy_indices:
            if self.noise_type == "gaussian":
                image = add_gaussian_noise(image.numpy().transpose(1, 2, 0)).transpose(2, 0, 1)
                image = torch.tensor(image, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, label

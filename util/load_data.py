import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from util.noise_dataset import MixedDataset

def get_subset(subset='Train', data_dir='Deepfake_Dataset', data_percentage=1.0, noise=False):
    subset_path = os.path.join(data_dir, subset)
    
    # Load the full dataset
    if noise:
        dataset = MixedDataset(root_dir=subset_path, noise_fraction=0.5, noise_type="gaussian")
    else:
        dataset = datasets.ImageFolder(root=subset_path)

    if data_percentage < 1.0:
        # Calculate the number of samples to use
        num_samples = int(len(dataset) * data_percentage)
        # Randomly select indices for the subset
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        # Create a subset of the dataset
        dataset = Subset(dataset, indices)

    return dataset


def load_dataset(dataset, batch_size=32, shuffle=True):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if isinstance(dataset, Subset):
        dataset.dataset.transform = transform
    else:
        dataset.transform = transform

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

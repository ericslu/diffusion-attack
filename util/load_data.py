import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_dataset(subset='Train', data_dir='Deepfake_Dataset', batch_size=32, shuffle=True, data_percentage=1.0):
    """
    Loads a subset of the dataset.

    Args:
        subset (str): One of 'Train', 'Validation', or 'Test'.
        data_dir (str): Path to the root dataset directory.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        data_percentage (float): Percentage of data to load (between 0 and 1).

    Returns:
        DataLoader: PyTorch DataLoader for the specified subset.
    """
    # Define data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    subset_path = os.path.join(data_dir, subset)
    
    # Load the full dataset
    dataset = datasets.ImageFolder(root=subset_path, transform=transform)
    
    # Use a subset of the dataset if data_percentage < 1.0
    if data_percentage < 1.0:
        # Calculate the number of samples to use
        num_samples = int(len(dataset) * data_percentage)
        # Randomly select indices for the subset
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        # Create a subset of the dataset
        dataset = Subset(dataset, indices)
    
    # Create a DataLoader for the dataset (or subset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

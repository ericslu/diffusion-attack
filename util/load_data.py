import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(subset='Train', batch_size=32, shuffle=True, data_dir='Deepfake_Dataset'):
    """
    Loads the dataset
    Args:
        subset (str): One of 'Train', 'Validation', or 'Test'.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        data_dir (str): Path to the root dataset directory.

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
    
    dataset = datasets.ImageFolder(root=subset_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

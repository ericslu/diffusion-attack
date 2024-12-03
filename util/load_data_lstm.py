import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class VideoSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length, transform=None):
        """
        Dataset for loading video sequences as tensors.

        Args:
            root_dir (str): Root directory containing video sequences.
            sequence_length (int): Number of frames per sequence.
            transform (callable, optional): Transform to be applied to each frame.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = []

        for sequence_folder in os.listdir(root_dir):
            sequence_path = os.path.join(root_dir, sequence_folder)

            # Skip non-directory items
            if not os.path.isdir(sequence_path):
                continue

            # Check if the directory contains exactly `sequence_length` valid files
            frames = [
                f for f in os.listdir(sequence_path)
                if os.path.isfile(os.path.join(sequence_path, f))
            ]
            if len(frames) == sequence_length:
                self.sequences.append(sequence_path)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_path = self.sequences[idx]
        
        # Sort and filter for valid files only
        frames = sorted(
            [f for f in os.listdir(sequence_path) if os.path.isfile(os.path.join(sequence_path, f))]
        )
        images = []

        for frame in frames:
            frame_path = os.path.join(sequence_path, frame)
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # Stack images into a tensor of shape (sequence_length, C, H, W)
        return torch.stack(images)

def load_dataset(data_dir, sequence_length, batch_size, transform=None):
    """
    Load the dataset and create a DataLoader.

    Args:
        data_dir (str): Directory containing sequences.
        sequence_length (int): Number of frames per sequence.
        batch_size (int): Batch size for DataLoader.
        transform (callable, optional): Transform to be applied to each frame.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = VideoSequenceDataset(root_dir=data_dir, sequence_length=sequence_length, transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No valid sequences found in {data_dir}. Check sequence length and dataset structure.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

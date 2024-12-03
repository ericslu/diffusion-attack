import os
import random
import torch

def create_random_sequences(data_loader, sequence_length=5):
    sequences = []
    labels = []

    all_images = []
    all_labels = []

    # Flatten all images and labels from the data loader
    for images, lbls in data_loader:
        all_images.extend(images)
        all_labels.extend(lbls)

    # Randomly shuffle the data
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    # Create sequences
    for i in range(0, len(all_images) - sequence_length + 1, sequence_length):
        sequence_images = all_images[i:i + sequence_length]
        sequence_label = all_labels[i + sequence_length - 1]  # Use the label of the last image
        sequences.append(torch.stack(sequence_images))
        labels.append(sequence_label)

    return torch.stack(sequences), torch.tensor(labels)

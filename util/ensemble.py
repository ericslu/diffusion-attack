#discrete cosine transformation, saliency partition

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import os
import util.saliency_partitioning
import torch.optim as optim
import torch.nn as nn

def ensemble_voting(predictions):
    return 1 if sum(predictions) > len(predictions) / 2 else 0


def get_scores(ensemble_models, dataloaders, device):
    total_correct = 0
    total_samples = 0
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    # Iterate through dataloaders for each ensemble model
    for model, dataloader in zip(ensemble_models, dataloaders):
        model.eval()  # Set to eval mode
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                predictions = torch.sigmoid(outputs) > 0.5
                loss = criterion(outputs, labels.float())

                total_correct += (predictions == labels).sum().item()
                total_loss += loss.item()
                total_samples += labels.size(0)

    accuracy = total_correct / total_samples * 100
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss
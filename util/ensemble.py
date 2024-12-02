#discrete cosine transformation, saliency partition
import cv2
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

def compute_dct(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply DCT
    dct = cv2.dct(np.float32(gray))
    return dct

def partition_dct(dct, num_partitions=4):
    h, w = dct.shape
    partitions = []
    step_h, step_w = h // 2, w // 2  # Example: 2x2 grid for 4 partitions
    for i in range(2):  # Rows
        for j in range(2):  # Columns
            partitions.append(dct[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w])
    return partitions

def ensemble_voting(predictions):
    return 1 if sum(predictions) > len(predictions) / 2 else 0

class DCTPartitionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, num_partitions=4):
        """
        Args:
            dataset: Original dataset (e.g., CIFAR-10, custom dataset).
            transform: Transformations for the models.
            num_partitions: Number of partitions for the DCT spectrum.
        """
        self.dataset = dataset
        self.transform = transform
        self.num_partitions = num_partitions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Convert image tensor to numpy for DCT computation
        image = image.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        image = (image * 255).astype(np.uint8)  # Rescale to 0-255

        dct = compute_dct(image)

        partitions = partition_dct(dct, self.num_partitions)


        if self.transform:
            partitions = [self.transform(part.astype(np.uint8)) for part in partitions]

        return partitions, label

def get_scores(ensemble_models, partitioned_dataloader):
    # Initialize accumulators for metrics
    total_correct = 0
    total_samples = 0
    total_loss = 0

    # Iterate through the test DataLoader
    for batch_partitions, labels in partitioned_dataloader:
        batch_predictions = []
        batch_losses = []
        
        for partitions, label in zip(batch_partitions, labels):  # Process each image in the batch
            predictions = []
            logits = []  # To accumulate raw outputs for loss calculation
            
            for model, partition in zip(ensemble_models, partitions):  # Each model gets one partition
                model.eval()
                with torch.no_grad():
                    partition = partition.unsqueeze(0).to(device)  # Add batch dimension and move to device
                    output = model(partition)
                    logits.append(output)  # Collect raw outputs
                    predictions.append(torch.sigmoid(output).item() > 0.5)
            
            # Ensemble voting for a single image
            final_prediction = ensemble_voting(predictions)
            batch_predictions.append(final_prediction)
            
            # Compute the loss for the current image
            logits_tensor = torch.cat(logits, dim=0)  # Combine logits from partitions
            average_logits = logits_tensor.mean()  # Take the mean across partitions
            label_tensor = label.unsqueeze(0).float().to(device)  # Prepare label tensor
            loss = binary_cross_entropy_with_logits(average_logits, label_tensor)
            batch_losses.append(loss.item())

        # Update metrics
        total_correct += sum((torch.tensor(batch_predictions) == labels).cpu().numpy())
        total_samples += len(labels)
        total_loss += sum(batch_losses)
    return total_correct / total_samples * 100, total_loss / total_samples
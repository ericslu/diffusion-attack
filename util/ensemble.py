#discrete cosine transformation, saliency partition

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
import os
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
from torchvision.transforms import ToTensor

def ensemble_voting(predictions):
    return 1 if sum(predictions) > len(predictions) / 2 else 0

def preprocess_images(dataset, save_dir, num_partitions=4):
    """
    Preprocess dataset by creating partitioned images and saving them.
    Args:
        dataset: PyTorch Dataset with images.
        save_dir: Base directory to save preprocessed partitioned images.
        num_partitions: Number of frequency partitions (default 4).
    """
    def ensure_unique_directory(base_dir):
        """Create a unique directory if one with the base name already exists."""
        if not os.path.exists(base_dir):
            return base_dir
        count = 1
        while os.path.exists(f"{base_dir}_{count}"):
            count += 1
        unique_dir = f"{base_dir}_{count}"
        os.makedirs(unique_dir, exist_ok=True)
        return unique_dir

    save_dir = ensure_unique_directory(save_dir)

    def compute_dct(image):
        return np.stack([dct(dct(channel.T, norm='ortho').T, norm='ortho') for channel in image], axis=-1)

    def compute_idct(dct_coefficients):
        return np.stack([idct(idct(channel.T, norm='ortho').T, norm='ortho') for channel in dct_coefficients], axis=-1)

    def create_partition_images(image, num_partitions):
        """Create partitioned images based on DCT coefficients."""
        dct_coefficients = compute_dct(image)
        h, w, c = dct_coefficients.shape
        partition_size = h // num_partitions
        partitions = []

        # We will now mask the low and high frequency parts
        for i in range(num_partitions):
            mask = np.zeros_like(dct_coefficients)

            # Define a more meaningful partitioning based on frequency importance
            # Keep the top (i+1) low-frequency bands
            start = 0
            end = (i + 1) * (h // num_partitions)

            mask[start:end, start:end, :] = 1  # Keep these frequency components

            # Apply the mask to the DCT coefficients
            masked_dct = dct_coefficients * mask

            # Inverse DCT to get back the partitioned image
            partitioned_image = compute_idct(masked_dct)
            
            # Clip the pixel values to be in the range [0, 255] and convert to uint8
            partitioned_image = np.clip(partitioned_image, 0, 255).astype(np.uint8)
            
            # Ensure the partition has the correct shape [H, W, C]
            partitioned_image = partitioned_image.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            
            # Append the partitioned image
            partitions.append(partitioned_image)
        return partitions

    for idx, (image, label) in enumerate(dataset):
        # Convert image to NumPy for processing
        image = image.permute(1, 2, 0).numpy() * 255  # [C, H, W] -> [H, W, C]

        partitions = create_partition_images(image, num_partitions)
        for i, part in enumerate(partitions):
            # Ensure dtype is uint8
            part = part.astype(np.uint8)

            # Check shape before saving
            if part.shape[2] != 3:  # For RGB images, the shape should be (H, W, 3)
                part = np.stack([part] * 3, axis=-1)  # Convert grayscale to RGB if necessary

            part_image = Image.fromarray(part)
            label_dir = "Real" if label == 1 else "Fake"
            part_dir = os.path.join(save_dir, f"partition_{i}", label_dir)
            os.makedirs(part_dir, exist_ok=True)
            part_filename = f"img_{idx}.png"
            part_image.save(os.path.join(part_dir, part_filename))

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(dataset)} images.")


def get_scores(ensemble_models, partitioned_dataloader, device):
    # Initialize accumulators for metrics
    total_correct = 0
    total_samples = 0
    total_loss = 0

    # Iterate through the test DataLoader
    for batch_partitions, labels in partitioned_dataloader:
        print(f"Batch Partitions: {[type(part) for part in batch_partitions[0]]}")  # Type of partitions
        print(f"Labels: {labels}")  # Type and content of labels
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
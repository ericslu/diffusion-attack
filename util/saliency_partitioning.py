import numpy as np
import torch
from scipy.fft import dct, idct
from util.cw_attack import CarliniWagnerL2Attack
import cv2

# Load model (Assuming you have a pretrained model like ResNet50)
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # Set model to evaluation mode
    return model

# Apply 2D DCT to an image
def dct2(image):
    return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

# Apply inverse DCT to a transformed image
def idct2(image):
    return idct(idct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

# Function to compute the saliency for a frequency band
def compute_saliency(model, image, freq_band, attack_steps=1000, epsilon=0.01):
    # Apply the frequency band mask (frequency partitioning)
    perturbed_image = apply_frequency_mask(image, freq_band)
    perturbed_image_tensor = torch.tensor(perturbed_image).float().unsqueeze(0).permute(0, 3, 1, 2).cpu()
    perturbed_image_tensor.requires_grad = True
    
    # Perform Carlini-Wagner attack on the perturbed image
    attack = CarliniWagnerL2Attack(model, learning_rate=epsilon)
    delta_x = attack.perturb(perturbed_image_tensor)  # Get perturbation
    
    # Compute gradients of the model with respect to perturbed image
    output = model(perturbed_image_tensor)  # Forward pass
    loss = output.sum()  # Sum of all outputs (can change based on specific task)
    loss.backward()  # Backward pass to compute gradients
    
    # Gradient with respect to the input image
    gradient = perturbed_image_tensor.grad.detach().cpu().numpy()
    saliency = np.sum(gradient * delta_x.cpu().numpy())  # Compute saliency as dot product
    return saliency

# Function to generate frequency bands
def generate_frequency_bands(image, num_bands=5):
    """
    Generate frequency bands for the DCT coefficients of an image.
    The number of bands is specified by `num_bands`.
    """
    dct_coeffs = dct2(image)  # Convert image to frequency domain
    
    # Define frequency bands by slicing the DCT coefficients array
    band_shape = dct_coeffs.shape
    band_size = band_shape[0] * band_shape[1] // num_bands
    
    # Create frequency bands by slicing the DCT coefficients
    frequencies = []
    for i in range(num_bands):
        start_idx = i * band_size
        end_idx = (i + 1) * band_size if i < num_bands - 1 else band_shape[0] * band_shape[1]
        band = np.unravel_index(np.arange(start_idx, end_idx), band_shape)
        frequencies.append(band)
    
    return frequencies

# Apply partition mask (simulate frequency filtering)
def apply_partition_mask(image, partition, frequencies):
    """
    Apply a mask to the DCT coefficients based on the partition.
    `partition` is a list of frequency indices that belong to this partition.
    `frequencies` is a list of frequency ranges or bands generated earlier.
    """
    dct_coeffs = dct2(image)  # Convert image to frequency domain
    
    # Create a mask based on the partition indices
    mask = np.zeros_like(dct_coeffs)
    
    for band_idx in partition:
        band = frequencies[band_idx]
        mask[band] = 1  # Set the mask for this frequency band
    
    # Apply the mask and return the inverse DCT of the modified coefficients
    return idct2(dct_coeffs * mask)

# Apply frequency band mask (simulate frequency filtering)
def apply_frequency_mask(image, freq_band):
    dct_coeffs = dct2(image)  # Convert image to frequency domain
    mask = np.zeros_like(dct_coeffs)
    mask[freq_band] = 1  # Only keep the frequencies in the given band
    return idct2(dct_coeffs * mask)  # Inverse DCT after masking frequencies

# Function to compute and sort saliencies for all frequencies
def compute_frequency_saliencies(model, image, frequencies, attack_steps=1000, epsilon=0.01):
    saliencies = []
    
    for i, freq_band in enumerate(frequencies):
        saliency = compute_saliency(model, image, freq_band, attack_steps, epsilon)
        saliencies.append((i, saliency))  # Store the index and saliency value
    
    # Sort frequencies by saliency in descending order
    sorted_saliencies = sorted(saliencies, key=lambda x: x[1], reverse=True)
    return sorted_saliencies

# Function to partition frequencies using round-robin
def partition_frequencies(sorted_saliencies, num_models):
    partitions = [[] for _ in range(num_models)]
    
    # Round-robin assignment of frequencies to models
    for idx, (freq_idx, _) in enumerate(sorted_saliencies):
        partitions[idx % num_models].append(freq_idx)
    
    return partitions

# Example: Distribute frequencies for an image
def partition_image_for_ensemble(model, image, num_models=4, attack_steps=1000, epsilon=0.01):
    # Assume frequencies are generated or pre-determined, each frequency band could be a range of DCT coefficients
    frequencies = generate_frequency_bands(image)  # Placeholder for generating frequency bands
    
    # Compute saliencies for each frequency
    sorted_saliencies = compute_frequency_saliencies(model, image, frequencies, attack_steps, epsilon)
    
    # Partition frequencies in round-robin fashion
    partitions = partition_frequencies(sorted_saliencies, num_models)
    
    # Create partitioned images for each model
    partitioned_images = []
    for partition in partitions:
        partitioned_image = apply_partition_mask(image, partition, frequencies)
        partitioned_images.append(partitioned_image)
    
    return partitioned_images

def load_image(image_path):
    image = cv2.imread(image_path)  # Load image in color (BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (224, 224))  # Resize to 224x224 for consistency
    
    return image

# # Example: Load an image and model, then partition the image
# image = load_image('path_to_image')  # Load an image as a numpy array
# model = load_model('path_to_model')  # Load a pre-trained model

# partitioned_images = partition_image_for_ensemble(model, image)

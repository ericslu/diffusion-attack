import numpy as np
import torch
from scipy.fft import dct, idct
from util.cw_attack import CarliniWagnerL2Attack
from util.fgsm_attack import FGSM
import cv2

def dct2(image):
    return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(image):
    return idct(idct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

def generate_radial_frequency_bands(image_shape, num_bands):
    h, w = image_shape[:2]
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    max_dist = distance.max()
    band_edges = np.linspace(0, max_dist, num_bands + 1)
    
    bands = []
    for i in range(num_bands):
        band_mask = (distance >= band_edges[i]) & (distance < band_edges[i + 1])
        bands.append(band_mask)
    
    return bands

def apply_frequency_mask(image, mask):
    dct_coeffs = dct2(image)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)
    masked_dct = dct_coeffs * mask
    return idct2(masked_dct)

def compute_saliency(model, image, freq_mask, attack, device='cpu'):
    perturbed_image = apply_frequency_mask(image, freq_mask)
    perturbed_image_tensor = (
        torch.tensor(perturbed_image)
        .float()
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
        .to(device)
    )
    perturbed_image_tensor.requires_grad = True

    delta_x = attack.perturb(perturbed_image_tensor)

    output = model(perturbed_image_tensor)
    loss = output.sum()
    loss.backward()
    gradient = perturbed_image_tensor.grad.detach().cpu().numpy()
    delta_x_np = delta_x.cpu().numpy()
    saliency = np.mean(gradient * delta_x_np)
    return saliency

# Compute saliencies for all frequency bands
def compute_frequency_saliencies(model, image, bands, attack_steps=1000, epsilon=0.01, device='cpu', attack_type = "fgsm"):
    if attack_type == "cw":
        attack = CarliniWagnerL2Attack(model, device, learning_rate=epsilon, max_iter=attack_steps)
    elif attack_type == "fgsm":
        attack = FGSM(model, epsilon=epsilon)
    saliencies = []
    for i, band_mask in enumerate(bands):
        saliency = compute_saliency(model, image, band_mask, attack, device)
        saliencies.append((i, saliency))
    return sorted(saliencies, key=lambda x: x[1], reverse=True)

def partition_frequencies(sorted_saliencies, num_models):
    partitions = [[] for _ in range(num_models)]
    for idx, (freq_idx, _) in enumerate(sorted_saliencies):
        partitions[idx % num_models].append(freq_idx)
    return partitions

def apply_partition(image, partitions, bands):
    partitioned_images = []
    for partition in partitions:
        mask = np.zeros(image.shape[:2])
        for band_idx in partition:
            mask += bands[band_idx]
        mask = np.expand_dims(mask, axis=-1)
        partitioned_image = apply_frequency_mask(image, mask)
        partitioned_images.append(partitioned_image)
    return partitioned_images

def partition_image_for_ensemble(model, image, num_models=3, num_bands=20, attack_steps=100, epsilon=0.01, device='cpu', attack_type = "fgsm"):
    bands = generate_radial_frequency_bands(image.shape, num_bands)
    sorted_saliencies = compute_frequency_saliencies(model, image, bands, attack_steps, epsilon, device, attack_type)
    partitions = partition_frequencies(sorted_saliencies, num_models)
    return apply_partition(image, partitions, bands)

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (224, 224))

import os
import shutil
import random

def create_subset_from_existing_dataset(root_dir, output_dir, subset_percentage=0.1):
    """
    Create a subset dataset by randomly selecting a percentage of images
    from each category (Real, Fake) in Train, Validation, and Test folders.
    
    :param root_dir: Path to the root of the original dataset
    :param output_dir: Path to the output directory where the subset will be saved
    :param subset_percentage: Percentage of data to include in the subset (e.g., 0.1 for 10%)
    """
    if not (0 < subset_percentage <= 1):
        raise ValueError("Subset percentage must be between 0 and 1.")
    splits = ['Train', 'Validation', 'Test']
    categories = ['Real', 'Fake']
    
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        for category in categories:
            original_dir = os.path.join(root_dir, split, category)
            if not os.path.exists(original_dir):
                print(f"Skipping {original_dir}, directory doesn't exist.")
                continue

            image_paths = [os.path.join(original_dir, fname) for fname in os.listdir(original_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]

            num_images_to_select = int(len(image_paths) * subset_percentage)

            selected_images = random.sample(image_paths, num_images_to_select)

            subset_category_dir = os.path.join(output_dir, split, category)
            os.makedirs(subset_category_dir, exist_ok=True)

            for img_path in selected_images:
                shutil.copy(img_path, subset_category_dir)
    
    print(f"Subset dataset created at {output_dir}")

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
    # Validate that the sum of percentages is 1 (if needed)
    if not (0 < subset_percentage <= 1):
        raise ValueError("Subset percentage must be between 0 and 1.")
    
    # Folder structure in original dataset (Train, Validation, Test) and categories (Real, Fake)
    splits = ['Train', 'Validation', 'Test']
    categories = ['Real', 'Fake']
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the dataset directories: Train, Validation, Test
    for split in splits:
        for category in categories:
            # Original directory for the current split and category
            original_dir = os.path.join(root_dir, split, category)
            
            # Skip if directory doesn't exist
            if not os.path.exists(original_dir):
                print(f"Skipping {original_dir}, directory doesn't exist.")
                continue

            # Get the list of all images in the category
            image_paths = [os.path.join(original_dir, fname) for fname in os.listdir(original_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]
            
            # Number of images to select based on the subset percentage
            num_images_to_select = int(len(image_paths) * subset_percentage)
            
            # Randomly select images
            selected_images = random.sample(image_paths, num_images_to_select)
            
            # Create output directory for the subset
            subset_category_dir = os.path.join(output_dir, split, category)
            os.makedirs(subset_category_dir, exist_ok=True)
            
            # Copy the selected images to the new subset directory
            for img_path in selected_images:
                shutil.copy(img_path, subset_category_dir)
    
    print(f"Subset dataset created at {output_dir}")

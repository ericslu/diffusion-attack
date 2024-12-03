import os
import random
import shutil

def split_data(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into Train, Validation, and Test sets.

    Args:
        source_dir (str): Directory containing all data.
        train_dir (str): Directory to save the training data.
        val_dir (str): Directory to save the validation data.
        test_dir (str): Directory to save the testing data.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
    """
    # Ensure destination directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get the list of videos/sequences and shuffle them
    videos = os.listdir(source_dir)
    random.shuffle(videos)

    # Calculate splits
    train_split = int(train_ratio * len(videos))
    val_split = int(val_ratio * len(videos)) + train_split

    for i, video in enumerate(videos):
        src = os.path.join(source_dir, video)
        
        # Determine destination
        if i < train_split:
            dst = os.path.join(train_dir, video)
        elif i < val_split:
            dst = os.path.join(val_dir, video)
        else:
            dst = os.path.join(test_dir, video)

        # Skip if destination already exists
        if os.path.exists(dst):
            print(f"Skipping existing directory: {dst}")
            continue

        # Copy directory
        try:
            shutil.copytree(src, dst)
            print(f"Copied: {src} -> {dst}")
        except Exception as e:
            print(f"Error copying {src} to {dst}: {e}")

if __name__ == "__main__":
    # Example usage: Replace with your actual paths
    source_dir = "./LSTMDataset/All/Real"
    train_dir = "./LSTMDataset/Train/Real"
    val_dir = "./LSTMDataset/Validation/Real"
    test_dir = "./LSTMDataset/Test/Real"

    print("Splitting Real sequences...")
    split_data(source_dir, train_dir, val_dir, test_dir)

    source_dir = "./LSTMDataset/All/Fake"
    train_dir = "./LSTMDataset/Train/Fake"
    val_dir = "./LSTMDataset/Validation/Fake"
    test_dir = "./LSTMDataset/Test/Fake"

    print("Splitting Fake sequences...")
    split_data(source_dir, train_dir, val_dir, test_dir)

    print("Dataset splitting complete!")

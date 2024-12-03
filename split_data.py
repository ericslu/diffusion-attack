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
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    videos = os.listdir(source_dir)
    random.shuffle(videos)

    train_split = int(train_ratio * len(videos))
    val_split = int(val_ratio * len(videos)) + train_split

    for i, video in enumerate(videos):
        src = os.path.join(source_dir, video)
        if i < train_split:
            dst = os.path.join(train_dir, video)
        elif i < val_split:
            dst = os.path.join(val_dir, video)
        else:
            dst = os.path.join(test_dir, video)
        shutil.copytree(src, dst)

# Example Usage
# split_data(
#     source_dir="Dataset/Train/Real",
#     train_dir="Dataset/Train/Real",
#     val_dir="Dataset/Validation/Real",
#     test_dir="Dataset/Test/Real"
# )

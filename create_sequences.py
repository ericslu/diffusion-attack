import os
import shutil

def create_sequences(frame_dir, output_dir, sequence_length=5):
    """
    Organize extracted frames into sequences.

    Args:
        frame_dir (str): Directory containing extracted frames.
        output_dir (str): Directory to save organized sequences.
        sequence_length (int): Number of frames per sequence.
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_folder in os.listdir(frame_dir):
        video_path = os.path.join(frame_dir, video_folder)
        if not os.path.isdir(video_path):
            continue

        frames = sorted(os.listdir(video_path))
        if len(frames) < sequence_length:
            print(f"Skipping {video_folder} (not enough frames for a sequence).")
            continue

        video_output_dir = os.path.join(output_dir, video_folder)
        os.makedirs(video_output_dir, exist_ok=True)

        for i in range(0, len(frames) - sequence_length + 1, sequence_length):
            sequence_dir = os.path.join(video_output_dir, f"sequence_{i // sequence_length:03d}")
            os.makedirs(sequence_dir, exist_ok=True)
            for j in range(sequence_length):
                frame_path = os.path.join(video_path, frames[i + j])
                shutil.copy(frame_path, os.path.join(sequence_dir, f"frame_{j:03d}.jpg"))
    print('Done!')

# Example Usage
"""
frame_dir = "./Frames/Fake"
output_dir = "./LSTMDataset/Train/Fake"
create_sequences(frame_dir, output_dir, sequence_length=5)
"""

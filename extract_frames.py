import os
import cv2
from pathlib import Path

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at a specified frame rate.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the extracted frames.
        frame_rate (int): Number of frames to extract per second.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    video_frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps / frame_rate)
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(video_frames_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
        success, frame = cap.read()
    cap.release()
    print(f"Frames saved in {video_frames_dir}")

# Example Usage
"""
video_path = "./Videos/Fake/01_02__exit_phone_room__YVGY8LOK.mp4"
output_dir = "./Frames/Fake"
extract_frames(video_path, output_dir, frame_rate=1)
"""


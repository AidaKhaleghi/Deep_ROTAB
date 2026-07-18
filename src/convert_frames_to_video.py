import cv2
import os
from pathlib import Path
import argparse
import sys

def frames_to_video(frames_path, output_path, fps=30):
    frames_path = Path(frames_path)

    if not frames_path.exists():
        print(f"Error: The specified frames path '{frames_path}' does not exist.")
        return False
    
    frame_files = sorted([
        f for f in os.listdir(frames_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])

    if not frame_files:
        print(f"Error: No frame files found in the specified frames path '{frames_path}'.")
        return False
    
    print(f"Found {len(frame_files)} frame files in '{frames_path}'.")

    first_frame_path = os.path.join(frames_path, frame_files[0])
    first_frame = cv2.imread(first_frame_path)

    height, width = first_frame.shape[:2]
    print(f"Video resolution will be set to {width}x{height}.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open the video writer with path '{output_path}'.")
        return False
    
    print(f"Writing frames to video ...")
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_path, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: Could not read frame '{frame_path}'. Skipping.")
            continue
        
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)

        if (i + 1) % 10 == 0 or (i + 1) == len(frame_files):
            print(f"Processed {i + 1}/{len(frame_files)} frames.")
    
    out.release()
    print(f"Video saved successfully to '{output_path}'.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert video frames to a video file.'
    )
    parser.add_argument(
        'frames_path',
        help='Path to the folder containing video frames.'
    )
    parser.add_argument(
        '-o', '--output', default='output_video.mp4',
        help='Output video file path (default: output_video.mp4).'
    )
    parser.add_argument(
        '-fps', '--fps',
        type=int,
        default=30,
        help='Frames per second for the output video (default: 30).'
    )

    args = parser.parse_args()

    success = frames_to_video(
        args.frames_path,
        args.output,
        args.fps
    )

    sys.exit(0 if success else 1)

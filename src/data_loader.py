import os
import cv2
import numpy as np

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def load_video_frames(path, normalize=True, resize=None):
    # Check if the provided paths are valid directories
    if not os.path.isdir(path):
        raise ValueError(f"Provided path '{path}' not found or is not a directory.")

    # Sort the frames by name because our algorithm is online and we need to process frames in order
    all_frame_names = sorted(
        f for f in os.listdir(path)
        if f.lower().endswith(VALID_EXTENSIONS)
    )

    # Load frames and their names
    frames = []
    frame_names = []

    for frame_name in all_frame_names:
        frame_path = os.path.join(path, frame_name)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        if frame is None:
            print(f"Warning: Could not read image '{frame_path}'. Skipping.")
            continue
        
        # Resize the frame if a resize parameter is provided
        if resize is not None:
            frame = cv2.resize(frame, resize)
        
        # Normalize to [0, 1]
        if normalize:
            frame = (frame / 255.0).astype(np.float32)

        frames.append(frame)
        frame_names.append(frame_name[2:])  # Remove the first two characters from the frame name (e.g. in000001.png -> 000001.png)

    return frames, frame_names


def load_video_masks(path, resize=None):
    # Check if the provided paths are valid directories
    if not os.path.isdir(path):
       raise ValueError(f"Provided path '{path}' not found or is not a directory.")
    
    # Sort the masks by name like frames
    all_mask_names = sorted(
        f for f in os.listdir(path)
        if f.lower().endswith(VALID_EXTENSIONS)
    )

    # Load masks and their names
    masks = []
    mask_names = []

    for mask_name in all_mask_names:
        mask_path = os.path.join(path, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Could not read image '{mask_path}'. Skipping.")
            continue
        
        # Resize the mask if a resize parameter is provided
        if resize is not None:
            mask = cv2.resize(mask, resize)
        
        # Binarize the mask by thresholding: values > 127 become 1, else 0
        mask = (mask > 127).astype(np.float32)

        masks.append(mask)
        mask_names.append(mask_name[2:-4])  # Remove the first two characters from the mask name (e.g. gt000001.png -> 000001.png)

    return masks, mask_names


def load_temporal_roi(path):
    if not os.path.isfile(path):
        raise ValueError(f"Temporal ROI file '{path}' not found.")

    with open(path, "r") as f:
        first_line = f.readline().strip()

    parts = first_line.split()
    if len(parts) < 2:
        raise ValueError(f"Temporal ROI file '{path}' must contain two numbers on the first line, got: '{first_line}'")

    return int(parts[0]), int(parts[1])


def verify_frames_and_masks(frame_names, mask_names):
    if len(frame_names) != len(mask_names):
        print(f"Warning: Number of frames ({len(frame_names)}) does not match number of masks ({len(mask_names)}).")
        return False
    fn = [os.path.splitext(f)[0] for f in frame_names]
    mn = [os.path.splitext(m)[0] for m in mask_names]
    return fn == mn

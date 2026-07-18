import os
import cv2
import numpy as np

from data_loader import load_video_frames, load_video_masks, verify_frames_and_masks
from algorithm import ROTAB
from convert_frames_to_video import frames_to_video



# Configuration parameters
INPUT_FOLDER = "./datasets/"      # <-- set this to your frames folder
MASK_FOLDER = None        # <-- set this to your masks folder (optional)
DATASET_NAME = "highway"        # <-- set this to your dataset name (optional)
OUTPUT_FOLDER = "./results"    # <-- set this to where results go
RESIZE = None                             # e.g. (160, 120)

RANK = 5
MU = 0.1
ALPHA = 0.95
LAM_PRIME = 0.4
K = 5  # Number of initial frames to use for initialization

def to_uint8(img):
    return np.clip(img*255, 0, 255).astype(np.uint8)

def foreground_to_uint8(S):
    S_abs = np.abs(S)
    max_val = S_abs.max()
    if max_val > 1e-8:
        S_vis = (S_abs / max_val) * 255.0
    else:
        S_vis = S_abs
    return to_uint8(S_vis)

def make_side_by_side(original, background, foreground, mask=None):
    h = original.shape[0]
    divider = np.full((h, 4), 255, dtype=np.uint8)
    if mask is not None:
        return cv2.hconcat([
            to_uint8(original), divider,
            to_uint8(background), divider,
            foreground_to_uint8(foreground), divider,
            to_uint8(mask)
        ])
    return cv2.hconcat([
        to_uint8(original), divider,
        to_uint8(background), divider,
        foreground_to_uint8(foreground)
    ])

def main():
    final_output_folder = os.path.join(OUTPUT_FOLDER, DATASET_NAME)
    os.makedirs(final_output_folder, exist_ok=True)

    frames, frame_names = load_video_frames(INPUT_FOLDER, resize=RESIZE)
    if MASK_FOLDER:
        masks, mask_names = load_video_masks(MASK_FOLDER, resize=RESIZE)
        print(f"Loaded {len(frames)} frames and {len(masks)} masks.")
        print(f"Frame and mask names match: {verify_frames_and_masks(frame_names, mask_names)}")
    else:
        print(f"Loaded {len(frames)} frames. No masks provided.")
    
    model = ROTAB(
        init_frames=frames[:K],
        rank=RANK,
        mu=MU,
        alpha=ALPHA,
        lam_prime=LAM_PRIME
    )

    for i, (frame, frame_name) in enumerate(zip(frames[K:], frame_names[K:])):
        L, S = model.process_frame(frame)

        if MASK_FOLDER:
            mask = masks[i+K]
            side_by_side = make_side_by_side(frame, L, S, mask)
        else:
            side_by_side = make_side_by_side(frame, L, S)

        output_path = os.path.join(final_output_folder, f"result_{frame_name}")
        cv2.imwrite(output_path, side_by_side)
        print(f"Processed frame {i+K+1}/{len(frames)}: {output_path}")
    
    print("Processing complete. Results saved in:", final_output_folder)

    frames_to_video(final_output_folder, os.path.join(final_output_folder, f"{DATASET_NAME}.mp4"))
    print(f"Video created successfully at: {os.path.join(final_output_folder, f'{DATASET_NAME}.mp4')}")


if __name__ == "__main__":
    main()

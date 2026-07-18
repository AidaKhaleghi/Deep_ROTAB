import os
import cv2
import numpy as np

from data_loader import load_video_frames, load_video_masks, verify_frames_and_masks, load_temporal_roi
from algorithm import ROTAB
from convert_frames_to_video import frames_to_video
from metrics import binarize_foreground, compute_metrics
from visualize_metrics import save_metrics_csv, plot_metrics


# Configuration parameters
INPUT_FOLDER = r"datasets\CDNet\intermittentObjectMotion\sofa\input"      # <-- set this to your frames folder
MASK_FOLDER = r"datasets\CDNet\intermittentObjectMotion\sofa\groundtruth"        # <-- set this to your masks folder (optional)
TEMPORAL_ROI_PATH = r"datasets\CDNet\intermittentObjectMotion\sofa\temporalROI.txt"  # <-- CDNet temporalROI.txt; set to None to treat every gt frame as valid
DATASET_NAME = "sofa"        # <-- set this to your dataset name (optional)
OUTPUT_FOLDER = r".\results\v1_baseline"    # <-- set this to where results go
RESIZE = None                             # e.g. (160, 120)

RANK = 30
MU = 0.1
ALPHA = 0.95
LAM_PRIME = 0.04
K = 10  # Number of initial frames to use for initialization

DELTA = 0.1              # Threshold on |S| (normalized [0,1] scale) to binarize the foreground mask
METRIC_PRINT_INTERVAL = 100  # Print running metrics every N frames (only used when MASK_FOLDER is set)

DELETE_FRAMES_AFTER_VIDEO = True  # Set to True to delete frames after creating the video, False to keep them

def to_uint8(img):
    return np.clip(img*255, 0, 255).astype(np.uint8)

def s_to_uint8(S):
    return (binarize_foreground(S, DELTA) * 255).astype(np.uint8)

def make_side_by_side(original, background, foreground, mask=None):
    h = original.shape[0]
    divider = np.full((h, 4), 255, dtype=np.uint8)
    if mask is not None:
        return cv2.hconcat([
            to_uint8(original), divider,
            to_uint8(background), divider,
            s_to_uint8(foreground), divider,
            to_uint8(mask)
        ])
    return cv2.hconcat([
        to_uint8(original), divider,
        to_uint8(background), divider,
        s_to_uint8(foreground)
    ])

def main():
    final_output_folder = os.path.join(OUTPUT_FOLDER, DATASET_NAME)
    os.makedirs(final_output_folder, exist_ok=True)

    frames, frame_names = load_video_frames(INPUT_FOLDER, resize=RESIZE)
    mask_by_name = {}
    if MASK_FOLDER:
        masks, mask_names = load_video_masks(MASK_FOLDER, resize=RESIZE)
        mask_by_name = dict(zip(mask_names, masks))
        print(f"Loaded {len(frames)} frames and {len(masks)} masks.")
        print(f"Frame and mask names match: {verify_frames_and_masks(frame_names, mask_names)}")
    else:
        print(f"Loaded {len(frames)} frames. No masks provided.")

    roi_start, roi_end = None, None
    if MASK_FOLDER and TEMPORAL_ROI_PATH:
        roi_start, roi_end = load_temporal_roi(TEMPORAL_ROI_PATH)
        print(f"Temporal ROI: ground truth is valid for frames {roi_start} to {roi_end}.")

    model = ROTAB(
        init_frames=frames[:K],
        rank=RANK,
        mu=MU,
        alpha=ALPHA,
        lam_prime=LAM_PRIME
    )

    metric_frame_numbers, precisions, recalls, f1s = [], [], [], []

    for i, (frame, frame_name) in enumerate(zip(frames[K:], frame_names[K:])):
        L, S = model.process_frame(frame)

        gt_mask = mask_by_name.get(frame_name[:-4])
        if gt_mask is not None:
            frame_number = int(frame_name[:-4])
            gt_is_valid = roi_start is None or (roi_start <= frame_number <= roi_end)
            if gt_is_valid:
                pred_mask = binarize_foreground(S, DELTA)
                precision, recall, f1 = compute_metrics(pred_mask, gt_mask)
                metric_frame_numbers.append(frame_number)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                if (i + 1) % METRIC_PRINT_INTERVAL == 0:
                    print(f"  Frame {i+K+1} metrics -> precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
            side_by_side = make_side_by_side(frame, L, S, gt_mask)
        else:
            side_by_side = make_side_by_side(frame, L, S)

        output_path = os.path.join(final_output_folder, f"result_{frame_name}")
        cv2.imwrite(output_path, side_by_side)
        if (i + 1) % METRIC_PRINT_INTERVAL == 0:
            print(f"Processed frame {i+K+1}/{len(frames)}: {output_path}")

    frames_to_video(final_output_folder, os.path.join(final_output_folder, f"{DATASET_NAME}.mp4"), delete_frames=DELETE_FRAMES_AFTER_VIDEO)

    if precisions:
        print(f"Average metrics over {len(precisions)} frames with valid ground truth -> "
              f"precision: {np.mean(precisions):.4f}, recall: {np.mean(recalls):.4f}, f1: {np.mean(f1s):.4f}")

        metrics_csv_path = os.path.join(final_output_folder, "metrics.csv")
        save_metrics_csv(metrics_csv_path, metric_frame_numbers, precisions, recalls, f1s)
        print(f"Saved per-frame metrics to: {metrics_csv_path}")

        plot_metrics(metrics_csv_path, final_output_folder)

    print("Processing complete. Results saved in:", final_output_folder)


if __name__ == "__main__":
    main()

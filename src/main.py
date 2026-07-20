import os
import cv2
import numpy as np

from data_loader import load_video_frames, load_video_masks, verify_frames_and_masks, load_temporal_roi
from algorithm import ROTAB
from algorithm_deep_b import ROTABDeepB
from algorithm_deep_bs import ROTABDeepBS
from convert_frames_to_video import frames_to_video
from metrics import binarize_foreground, compute_metrics
from visualize_metrics import save_metrics_csv, plot_metrics, save_loss_csv, plot_loss


# Configuration parameters
INPUT_FOLDER = r"datasets\CDNet\baseline\pedestrians\input"      # <-- set this to your frames folder
MASK_FOLDER = r"datasets\CDNet\baseline\pedestrians\groundtruth"        # <-- set this to your masks folder (optional)
TEMPORAL_ROI_PATH = r"datasets\CDNet\baseline\pedestrians\temporalROI.txt"  # <-- CDNet temporalROI.txt; set to None to treat every gt frame as valid
DATASET_NAME = "pedestrians"        # <-- set this to your dataset name (optional)
MODEL_TYPE = "deep_bs"       # "baseline" (closed-form b, prox S), "deep_b" (ConvLSTM b), "deep_bs" (ConvLSTM b + conv S)
OUTPUT_FOLDER = {
    "baseline": r".\results\v1_baseline",
    "deep_b": r".\results\v2_deep_b",
    "deep_bs": r".\results\v5_deep_bs_new_loss",
}[MODEL_TYPE]
RESIZE = None                             # e.g. (160, 120)

RANK = 30
MU = 0.1
ALPHA = 0.95
LAM_PRIME = 0.04
K = 10  # Number of initial frames to use for initialization

# deep_b / deep_bs (network) settings
LR = 1e-3          # Online learning rate for the networks
TRAIN_STEPS = 1    # Gradient steps per frame
S_CHANNEL = 16     # deep_bs only: hidden channels of the S-network
S_LAYERS = 3       # deep_bs only: hidden conv layers of the S-network
LAM_TV = 0.03      # deep_bs only: weight of the total-variation loss on S
LAM_MOTION = 0.01  # deep_bs only: weight of the motion-gated sparsity loss
MOTION_TAU = 0.03  # deep_bs only: |D[t]-D[t-1]| below this = static pixel

DELTA = 0.10              # Threshold on |S| (normalized [0,1] scale) to binarize the foreground mask
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

    if MODEL_TYPE == "deep_bs":
        model = ROTABDeepBS(
            init_frames=frames[:K],
            rank=RANK,
            mu=MU,
            alpha=ALPHA,
            lam_prime=LAM_PRIME,
            lr=LR,
            train_steps=TRAIN_STEPS,
            s_channel=S_CHANNEL,
            s_layers=S_LAYERS,
            lam_tv=LAM_TV,
            lam_motion=LAM_MOTION,
            motion_tau=MOTION_TAU
        )
        print(f"Using ROTABDeepBS (ConvLSTM b + conv S) on device: {model.device}")
    elif MODEL_TYPE == "deep_b":
        model = ROTABDeepB(
            init_frames=frames[:K],
            rank=RANK,
            mu=MU,
            alpha=ALPHA,
            lam_prime=LAM_PRIME,
            lr=LR,
            train_steps=TRAIN_STEPS
        )
        print(f"Using ROTABDeepB (ConvLSTM b) on device: {model.device}")
    else:
        model = ROTAB(
            init_frames=frames[:K],
            rank=RANK,
            mu=MU,
            alpha=ALPHA,
            lam_prime=LAM_PRIME
        )

    metric_frame_numbers, precisions, recalls, f1s = [], [], [], []
    loss_frame_numbers, losses, loss_parts = [], [], []

    for i, (frame, frame_name) in enumerate(zip(frames[K:], frame_names[K:])):
        L, S = model.process_frame(frame)

        if getattr(model, "last_loss", None) is not None:
            loss_frame_numbers.append(i + K + 1)
            losses.append(model.last_loss)
            if getattr(model, "last_loss_parts", None) is not None:
                loss_parts.append(model.last_loss_parts)

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
            loss_info = f" | net loss: {model.last_loss:.6f}" if getattr(model, "last_loss", None) is not None else ""
            print(f"Processed frame {i+K+1}/{len(frames)}: {output_path}{loss_info}")

    frames_to_video(final_output_folder, os.path.join(final_output_folder, f"{DATASET_NAME}.mp4"), delete_frames=DELETE_FRAMES_AFTER_VIDEO)

    if precisions:
        print(f"Average metrics over {len(precisions)} frames with valid ground truth -> "
              f"precision: {np.mean(precisions):.4f}, recall: {np.mean(recalls):.4f}, f1: {np.mean(f1s):.4f}")

        metrics_csv_path = os.path.join(final_output_folder, "metrics.csv")
        save_metrics_csv(metrics_csv_path, metric_frame_numbers, precisions, recalls, f1s)
        print(f"Saved per-frame metrics to: {metrics_csv_path}")

        plot_metrics(metrics_csv_path, final_output_folder)

    if losses:
        parts = loss_parts if len(loss_parts) == len(losses) else None
        loss_csv_path = os.path.join(final_output_folder, "network_loss.csv")
        save_loss_csv(loss_csv_path, loss_frame_numbers, losses, parts=parts)
        print(f"Saved per-frame network loss to: {loss_csv_path}")
        plot_loss(loss_frame_numbers, losses, final_output_folder, parts=parts)

    print("Processing complete. Results saved in:", final_output_folder)


if __name__ == "__main__":
    main()

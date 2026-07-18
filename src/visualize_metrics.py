import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Fixed series colors (colorblind-safe ordering): precision, recall, f1
SERIES_COLORS = {
    "precision": "#2a78d6",  # blue
    "recall": "#1baf7a",     # aqua
    "f1": "#eda100",         # yellow
}
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#898781"


def load_metrics_csv(csv_path):
    frames, precisions, recalls, f1s = [], [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            precisions.append(float(row["precision"]))
            recalls.append(float(row["recall"]))
            f1s.append(float(row["f1"]))
    return np.array(frames), np.array(precisions), np.array(recalls), np.array(f1s)


def save_metrics_csv(csv_path, frame_numbers, precisions, recalls, f1s):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "precision", "recall", "f1"])
        for row in zip(frame_numbers, precisions, recalls, f1s):
            writer.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}"])


def rolling_mean(values, window):
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    # Pad the start so the smoothed series aligns with the frame axis
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, smoothed])


def _style_axes(ax):
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=AXIS_COLOR, labelsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_metrics_over_time(frames, precisions, recalls, f1s, output_path, window=None):
    if window is None:
        window = max(1, min(50, len(frames) // 10))

    fig, ax = plt.subplots(figsize=(10, 5))
    series = [("precision", precisions), ("recall", recalls), ("f1", f1s)]

    for name, values in series:
        color = SERIES_COLORS[name]
        ax.plot(frames, values, color=color, linewidth=0.8, alpha=0.25)
        smoothed = rolling_mean(values, window)
        ax.plot(frames, smoothed, color=color, linewidth=2, label=name)
        # Direct label at the end of the smoothed line
        last_valid = np.where(~np.isnan(smoothed))[0]
        if len(last_valid) > 0:
            idx = last_valid[-1]
            ax.annotate(f" {name} {smoothed[idx]:.2f}", (frames[idx], smoothed[idx]),
                        color=color, fontsize=9, va="center", fontweight="bold")

    _style_axes(ax)
    ax.set_xlabel("Frame number", color=AXIS_COLOR)
    ax.set_ylabel("Score", color=AXIS_COLOR)
    ax.set_title(f"Per-frame metrics (rolling mean, window={window})", fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_running_average(frames, precisions, recalls, f1s, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = np.arange(1, len(frames) + 1)
    series = [("precision", precisions), ("recall", recalls), ("f1", f1s)]

    for name, values in series:
        color = SERIES_COLORS[name]
        running = np.cumsum(values) / counts
        ax.plot(frames, running, color=color, linewidth=2, label=name)
        ax.annotate(f" {name} {running[-1]:.3f}", (frames[-1], running[-1]),
                    color=color, fontsize=9, va="center", fontweight="bold")

    _style_axes(ax)
    ax.set_xlabel("Frame number", color=AXIS_COLOR)
    ax.set_ylabel("Running average", color=AXIS_COLOR)
    ax.set_title("Running average of metrics over valid frames", fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_loss_csv(csv_path, frame_numbers, losses):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "loss"])
        for frame, loss in zip(frame_numbers, losses):
            writer.writerow([frame, f"{loss:.8f}"])


def plot_loss(frame_numbers, losses, output_folder, window=None):
    frames = np.array(frame_numbers)
    losses = np.array(losses)
    if len(frames) == 0:
        print("No loss values found; skipping loss plot.")
        return
    if window is None:
        window = max(1, min(50, len(frames) // 10))

    fig, ax = plt.subplots(figsize=(10, 5))
    color = SERIES_COLORS["precision"]
    ax.plot(frames, losses, color=color, linewidth=0.8, alpha=0.25)
    smoothed = rolling_mean(losses, window)
    ax.plot(frames, smoothed, color=color, linewidth=2, label="network loss")
    last_valid = np.where(~np.isnan(smoothed))[0]
    if len(last_valid) > 0:
        idx = last_valid[-1]
        ax.annotate(f" {smoothed[idx]:.2e}", (frames[idx], smoothed[idx]),
                    color=color, fontsize=9, va="center", fontweight="bold")

    _style_axes(ax)
    # Loss lives on its own scale (typically ~1e-4), not in [0, 1]
    ax.set_ylim(0, np.nanmax(losses) * 1.05)
    ax.set_xlabel("Frame number", color=AXIS_COLOR)
    ax.set_ylabel("Loss", color=AXIS_COLOR)
    ax.set_title(f"Online network loss per frame (rolling mean, window={window})", fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    output_path = os.path.join(output_folder, "network_loss.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss plot: {output_path}")


def plot_metrics(csv_path, output_folder, window=None):
    frames, precisions, recalls, f1s = load_metrics_csv(csv_path)
    if len(frames) == 0:
        print("No metric rows found in CSV; skipping plots.")
        return
    over_time_path = os.path.join(output_folder, "metrics_over_time.png")
    running_avg_path = os.path.join(output_folder, "metrics_running_average.png")
    plot_metrics_over_time(frames, precisions, recalls, f1s, over_time_path, window=window)
    plot_running_average(frames, precisions, recalls, f1s, running_avg_path)
    print(f"Saved plots: {over_time_path}, {running_avg_path}")

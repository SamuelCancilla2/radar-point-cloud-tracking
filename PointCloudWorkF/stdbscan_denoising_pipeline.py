#!/usr/bin/env python3
"""
ST-DBSCAN Radar Point Cloud Denoising Pipeline

This pipeline implements the methodology described in the abstract:
1. Transform raw radar data (Status, Scale, Range, Gain, Angle, Echo Values)
   into Cartesian coordinates and point cloud form
2. Apply ST-DBSCAN to cluster consistent object returns while suppressing noise
3. Produce correct object groupings across sequential frames
4. Reduce false positives caused by noise

Usage:
    python stdbscan_denoising_pipeline.py --max-frames 50
    python stdbscan_denoising_pipeline.py --eps-space 8.0 --min-samples 15
"""

from __future__ import annotations

import argparse
import gc
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import os

import numpy as np
import pandas as pd

# Limit numpy threads to avoid oversubscription with multiprocessing
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn required for ST-DBSCAN clustering")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    warnings.warn("Pillow not installed - GIF generation disabled. Install with: pip install Pillow")

# =============================================================================
# Configuration
# =============================================================================

# Radar parameters
ANGLE_SCALE = 360.0 / 8196.0  # Convert radar angle units to degrees
NUM_ECHO_COLUMNS = 1024

# Processing parameters
INTENSITY_THRESHOLD = 10.0    # Minimum intensity to keep a point
POINT_STRIDE = 4              # Keep every Nth point for efficiency
MAX_TIME_DIFF_MS = 2000       # Max time diff to consider same frame (ms)
MAX_WORKERS = min(4, os.cpu_count() or 1)  # Parallel workers for CSV loading

# ST-DBSCAN parameters (tunable via command line)
DEFAULT_EPS_SPACE = 8.0       # Spatial neighborhood radius (meters)
DEFAULT_EPS_TIME = 2.0        # Temporal neighborhood (frames)
DEFAULT_MIN_SAMPLES = 15      # Minimum points to form a cluster
DEFAULT_MIN_FRAMES = 2        # Minimum frames a cluster must span (filters transient noise)

# Visualization
PLOT_MAX_POINTS = 500_000     # Max points to plot


# =============================================================================
# Data Loading Functions
# =============================================================================

def parse_timestamp(filename: str) -> datetime:
    """Extract timestamp from filename format: YYYYMMDD_HHMMSS_mmm.csv"""
    match = re.match(r"(\d{8})_(\d{6})_(\d{3})", filename)
    if not match:
        raise ValueError(f"Cannot parse timestamp from: {filename}")
    date_str, time_str, ms_str = match.groups()
    dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    return dt.replace(microsecond=int(ms_str) * 1000)


def load_radar_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a radar CSV file and convert to Cartesian coordinates (optimized).

    Returns: (x, y, intensities) arrays
    """
    # Use numpy for faster loading of numeric data
    try:
        # Skip header row, load only what we need
        # Columns: Status(0), Scale(1), Range(2), Gain(3), Angle(4), Echo_0..Echo_1023(5:)
        data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=np.float32,
                             filling_values=0.0)
    except Exception:
        # Fallback to pandas for malformed files
        col_names = ["Status", "Scale", "Range", "Gain", "Angle"] + \
                    [f"Echo_{i}" for i in range(NUM_ECHO_COLUMNS)]
        df = pd.read_csv(path, header=None, names=col_names, skiprows=1, engine="c")
        if df.empty:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        data = df.to_numpy(dtype=np.float32)

    if data.size == 0 or data.ndim != 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Extract columns efficiently (no copy with slicing)
    angles_rad = np.deg2rad(data[:, 4] * ANGLE_SCALE)
    max_ranges = data[:, 1]  # Scale column
    echo_data = data[:, 5:]  # All echo columns

    # Pre-compute constants
    num_rows, num_bins = echo_data.shape
    bin_indices = np.arange(num_bins, dtype=np.float32)

    # Vectorized range calculation
    range_res = max_ranges / num_bins
    ranges = np.outer(range_res, bin_indices)

    # Vectorized polar to Cartesian (use broadcasting)
    cos_angles = np.cos(angles_rad)[:, np.newaxis]
    sin_angles = np.sin(angles_rad)[:, np.newaxis]
    x = ranges * cos_angles
    y = ranges * sin_angles

    # Filter by intensity threshold and flatten
    mask = echo_data > INTENSITY_THRESHOLD
    x_flat = x[mask]
    y_flat = y[mask]
    z_flat = echo_data[mask]

    # Apply stride for efficiency (use view when possible)
    if POINT_STRIDE > 1:
        x_flat = x_flat[::POINT_STRIDE]
        y_flat = y_flat[::POINT_STRIDE]
        z_flat = z_flat[::POINT_STRIDE]

    return x_flat, y_flat, z_flat


def discover_files(data_dir: Path) -> Dict[int, List[Tuple[datetime, Path]]]:
    """
    Discover radar CSV files organized by gain.
    Returns: {gain: [(timestamp, path), ...]}
    """
    gain_files: Dict[int, List[Tuple[datetime, Path]]] = {}

    for gain_dir in data_dir.iterdir():
        if not gain_dir.is_dir():
            continue
        match = re.match(r"gain[_-]?(\d+)", gain_dir.name, re.IGNORECASE)
        if not match:
            continue
        gain = int(match.group(1))

        files = []
        for csv_path in sorted(gain_dir.glob("*.csv")):
            try:
                ts = parse_timestamp(csv_path.name)
                files.append((ts, csv_path))
            except ValueError:
                continue

        if files:
            gain_files[gain] = sorted(files, key=lambda x: x[0])

    return gain_files


def group_into_frames(gain_files: Dict[int, List[Tuple[datetime, Path]]]) -> List[Dict[int, Path]]:
    """
    Group files from different gains into temporal frames.
    Files within MAX_TIME_DIFF_MS are considered the same frame.
    """
    all_files = []
    for gain, files in gain_files.items():
        for ts, path in files:
            all_files.append((ts, gain, path))

    all_files.sort(key=lambda x: x[0])

    frames = []
    current_frame: Dict[int, Path] = {}
    current_time = None

    for ts, gain, path in all_files:
        if current_time is None:
            current_time = ts
            current_frame[gain] = path
        elif (ts - current_time).total_seconds() * 1000 <= MAX_TIME_DIFF_MS:
            if gain not in current_frame:
                current_frame[gain] = path
        else:
            if current_frame:
                frames.append(current_frame)
            current_frame = {gain: path}
            current_time = ts

    if current_frame:
        frames.append(current_frame)

    return frames


def load_frame(frame: Dict[int, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all gain data for a single frame. Used for parallel processing."""
    frame_x, frame_y, frame_z = [], [], []
    for gain, path in frame.items():
        x, y, z = load_radar_csv(path)
        if len(x) > 0:
            frame_x.append(x)
            frame_y.append(y)
            frame_z.append(z)

    if frame_x:
        return np.concatenate(frame_x), np.concatenate(frame_y), np.concatenate(frame_z)
    return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)


def load_frames_parallel(frames: List[Dict[int, Path]], max_workers: int = MAX_WORKERS) -> List[dict]:
    """Load multiple frames in parallel for faster processing."""
    frames_data = [None] * len(frames)

    # Use ProcessPoolExecutor for CPU-bound CSV parsing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(load_frame, frame): idx
                         for idx, frame in enumerate(frames)}

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                fx, fy, fz = future.result()
                frames_data[idx] = {'x': fx, 'y': fy, 'z': fz}
            except Exception as e:
                print(f"  Warning: Failed to load frame {idx}: {e}")
                frames_data[idx] = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}

            completed += 1
            if completed % 20 == 0:
                print(f"  Loaded {completed}/{len(frames)} frames...")

    return frames_data


# =============================================================================
# ST-DBSCAN Implementation
# =============================================================================

def st_dbscan(coords: np.ndarray, times: np.ndarray,
              eps_space: float, eps_time: float, min_samples: int,
              min_frames: int = 2) -> np.ndarray:
    """
    Spatio-Temporal DBSCAN clustering (optimized).

    Points are neighbors if:
    1. Spatial distance <= eps_space
    2. Temporal distance <= eps_time

    A point is a core point only if:
    - It has >= min_samples neighbors satisfying above conditions
    - Those neighbors span >= min_frames different frames (temporal persistence)

    This filters out transient noise blobs that only appear in a single frame.

    Returns cluster labels (-1 = noise)
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for ST-DBSCAN")

    n = coords.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)

    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)

    # Build spatial index - use float32 for memory efficiency
    coords_f32 = coords.astype(np.float32, copy=False)
    tree = BallTree(coords_f32, leaf_size=40)

    # Query in batches for better cache locality
    BATCH_SIZE = 10000
    spatial_neighbors = []
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_neighbors = tree.query_radius(coords_f32[start:end], r=eps_space)
        spatial_neighbors.extend(batch_neighbors)

    # Pre-convert times to int for frame counting
    times_int = times.astype(np.int32, copy=False)
    times_f32 = times.astype(np.float32, copy=False)

    def is_core_point(idx, neighbors):
        """Check if point is a core point (enough neighbors spanning enough frames)."""
        if len(neighbors) < min_samples:
            return False
        # Count unique frames among neighbors
        neighbor_frames = times_int[neighbors]
        unique_frames = len(np.unique(neighbor_frames))
        return unique_frames >= min_frames

    def get_st_neighbors(idx):
        """Get spatio-temporal neighbors for a point."""
        spatial_nb = spatial_neighbors[idx]
        time_diffs = np.abs(times_f32[spatial_nb] - times_f32[idx])
        return spatial_nb[time_diffs <= eps_time]

    cluster_id = 0
    # Track which points are queued to avoid duplicates
    in_queue = np.zeros(n, dtype=bool)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # Get spatio-temporal neighbors
        neighbors = get_st_neighbors(i)

        # Check if this is a core point (enough neighbors + temporal persistence)
        if not is_core_point(i, neighbors):
            continue  # stays as noise (-1)

        # Start new cluster
        labels[i] = cluster_id

        # Use list as queue (simpler and safe)
        queue = list(neighbors)
        in_queue[neighbors] = True

        while queue:
            pt = queue.pop(0)

            if not visited[pt]:
                visited[pt] = True
                # Get neighbors for this point
                pt_neighbors = get_st_neighbors(pt)

                # Only expand if this is also a core point
                if is_core_point(pt, pt_neighbors):
                    # Add new neighbors to queue (only if not already queued)
                    for nb in pt_neighbors:
                        if not visited[nb] and not in_queue[nb]:
                            queue.append(nb)
                            in_queue[nb] = True

            if labels[pt] == -1:
                labels[pt] = cluster_id

        # Reset in_queue for next cluster
        in_queue[:] = False
        cluster_id += 1

    return labels


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_before_after(output_dir: Path,
                      raw_x: np.ndarray, raw_y: np.ndarray, raw_z: np.ndarray,
                      denoised_x: np.ndarray, denoised_y: np.ndarray,
                      denoised_z: np.ndarray, labels: np.ndarray) -> None:
    """Create before/after comparison showing denoising effect."""
    if not HAS_MPL:
        print("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subsample for plotting
    def subsample(x, y, z, max_pts):
        if len(x) <= max_pts:
            return x, y, z
        idx = np.random.choice(len(x), max_pts, replace=False)
        return x[idx], y[idx], z[idx]

    # Before (raw data)
    ax1 = axes[0]
    rx, ry, rz = subsample(raw_x, raw_y, raw_z, PLOT_MAX_POINTS)
    scatter1 = ax1.scatter(rx, ry, c=rz, cmap='viridis', s=0.5, alpha=0.5)
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_title(f"Raw Point Cloud\n({len(raw_x):,} points)")
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='Intensity')

    # After (denoised - clustered points only)
    ax2 = axes[1]
    dx, dy, dz = subsample(denoised_x, denoised_y, denoised_z, PLOT_MAX_POINTS)
    # Color by cluster
    dl = labels[np.random.choice(len(labels), len(dx), replace=False)] if len(dx) < len(labels) else labels[:len(dx)]
    scatter2 = ax2.scatter(dx, dy, c=dl, cmap='tab20', s=0.5, alpha=0.5)
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    ax2.set_title(f"ST-DBSCAN Denoised\n({len(denoised_x):,} points, {len(np.unique(labels[labels >= 0]))} clusters)")
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / "denoising_comparison.png", dpi=200)
    plt.close()
    print(f"Saved: denoising_comparison.png")


def plot_temporal_clusters(output_dir: Path, frames_data: List[dict],
                          labels: np.ndarray, frame_indices: np.ndarray) -> None:
    """Create visualization of clusters across time frames."""
    if not HAS_MPL:
        return

    # Select a few frames to visualize
    unique_frames = np.unique(frame_indices)
    sample_frames = unique_frames[::max(1, len(unique_frames) // 6)][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax_idx, frame_id in enumerate(sample_frames):
        if ax_idx >= len(axes):
            break

        ax = axes[ax_idx]
        mask = frame_indices == frame_id
        x = frames_data[frame_id]['x']
        y = frames_data[frame_id]['y']
        frame_labels = labels[mask]

        # Subsample if needed
        if len(x) > 50000:
            idx = np.random.choice(len(x), 50000, replace=False)
            x, y = x[idx], y[idx]
            frame_labels = frame_labels[idx]

        # Plot noise in gray, clusters in colors
        noise_mask = frame_labels == -1
        cluster_mask = ~noise_mask

        if noise_mask.any():
            ax.scatter(x[noise_mask], y[noise_mask], c='lightgray', s=0.3, alpha=0.3, label='Noise')
        if cluster_mask.any():
            ax.scatter(x[cluster_mask], y[cluster_mask], c=frame_labels[cluster_mask],
                      cmap='tab20', s=1, alpha=0.7)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Frame {frame_id}")
        ax.set_aspect('equal')

    plt.suptitle("ST-DBSCAN Clustering Across Time Frames", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_clusters.png", dpi=200)
    plt.close()
    print(f"Saved: temporal_clusters.png")


def plot_noise_reduction_stats(output_dir: Path, stats: dict) -> None:
    """Create visualization of noise reduction statistics."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart: noise vs signal
    ax1 = axes[0]
    sizes = [stats['noise_points'], stats['signal_points']]
    labels = [f"Noise\n({stats['noise_points']:,})", f"Signal\n({stats['signal_points']:,})"]
    colors = ['#ff6b6b', '#4ecdc4']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Point Classification")

    # Bar chart: reduction stats
    ax2 = axes[1]
    categories = ['Raw Points', 'Denoised Points', 'Clusters Found']
    values = [stats['total_points'], stats['signal_points'], stats['num_clusters']]
    bars = ax2.bar(categories, values, color=['#3498db', '#2ecc71', '#9b59b6'])
    ax2.set_ylabel("Count")
    ax2.set_title("Denoising Results")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                f'{val:,}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "noise_reduction_stats.png", dpi=200)
    plt.close()
    print(f"Saved: noise_reduction_stats.png")


def track_clusters_across_frames(frames_data: List[dict], labels: np.ndarray,
                                  frame_indices: np.ndarray, max_dist: float = 20.0) -> dict:
    """
    Track clusters across frames by matching centroids.
    Returns a mapping from (frame_id, local_cluster_id) -> global_tracked_id
    """
    unique_frames = np.unique(frame_indices)

    # Compute cluster centroids for each frame
    frame_clusters = {}  # frame_id -> {cluster_id: (centroid_x, centroid_y, num_points)}
    for frame_id in unique_frames:
        mask = frame_indices == frame_id
        frame_labels = labels[mask]
        frame_x = np.concatenate([frames_data[frame_id]['x']])
        frame_y = np.concatenate([frames_data[frame_id]['y']])

        cluster_ids = np.unique(frame_labels[frame_labels >= 0])
        centroids = {}
        for cid in cluster_ids:
            cmask = frame_labels == cid
            centroids[cid] = (frame_x[cmask].mean(), frame_y[cmask].mean(), cmask.sum())
        frame_clusters[frame_id] = centroids

    # Track clusters across frames using greedy centroid matching
    global_id_map = {}  # (frame_id, local_cluster_id) -> global_tracked_id
    next_global_id = 0
    prev_centroids = {}  # global_id -> (x, y)

    for frame_id in unique_frames:
        centroids = frame_clusters[frame_id]
        used_global_ids = set()

        # Match current clusters to previous frame's tracked clusters
        matches = []
        for local_id, (cx, cy, _) in centroids.items():
            best_global_id = None
            best_dist = max_dist

            for global_id, (px, py) in prev_centroids.items():
                if global_id in used_global_ids:
                    continue
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_global_id = global_id

            matches.append((local_id, best_global_id, cx, cy))

        # Assign global IDs
        new_prev_centroids = {}
        for local_id, best_global_id, cx, cy in matches:
            if best_global_id is not None and best_global_id not in used_global_ids:
                global_id = best_global_id
                used_global_ids.add(global_id)
            else:
                global_id = next_global_id
                next_global_id += 1

            global_id_map[(frame_id, local_id)] = global_id
            new_prev_centroids[global_id] = (cx, cy)

        prev_centroids = new_prev_centroids

    return global_id_map, next_global_id


def create_comparison_gif(output_dir: Path, frames_data: List[dict],
                          labels: np.ndarray, frame_indices: np.ndarray,
                          fps: int = 2) -> None:
    """
    Create an animated GIF showing side-by-side comparison of raw vs clustered data
    for each frame over time, with consistent cluster tracking and IDs.
    """
    if not HAS_MPL or not HAS_PIL:
        print("matplotlib and Pillow required for GIF generation")
        return

    unique_frames = np.unique(frame_indices)

    # Skip the first frame (often glitched)
    if len(unique_frames) > 1:
        unique_frames = unique_frames[1:]

    num_frames = len(unique_frames)

    if num_frames == 0:
        print("No frames to animate")
        return

    print(f"  Creating comparison GIF with {num_frames} frames (skipping first frame)...")
    print("    Tracking clusters across frames...")

    # Track clusters across frames
    global_id_map, total_tracked = track_clusters_across_frames(
        frames_data, labels, frame_indices
    )

    # Temporary directory for frame images
    temp_dir = output_dir / "_temp_frames"
    temp_dir.mkdir(exist_ok=True)

    # Calculate consistent axis limits across all frames
    all_x = np.concatenate([fd['x'] for fd in frames_data if len(fd['x']) > 0])
    all_y = np.concatenate([fd['y'] for fd in frames_data if len(fd['y']) > 0])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    # Add padding
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    # Create color lookup table for tracked clusters (consistent colors)
    cmap = plt.colormaps.get_cmap('tab20')
    # Generate enough distinct colors
    num_colors = max(20, total_tracked + 1)
    tracked_colors = {}
    for i in range(num_colors):
        tracked_colors[i] = cmap(i % 20)[:3]

    frame_paths = []

    for idx, frame_id in enumerate(unique_frames):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))  # Wider to fit legend outside

        # Get frame data
        frame_data = frames_data[frame_id]
        x = frame_data['x']
        y = frame_data['y']
        z = frame_data['z']

        # Get labels for this frame
        mask = frame_indices == frame_id
        frame_labels = labels[mask]

        # Left panel: Raw point cloud (colored by intensity)
        ax1 = axes[0]
        if len(x) > 0:
            z_norm = np.clip(z / z.max() if z.max() > 0 else z, 0, 1)
            scatter1 = ax1.scatter(x, y, c=z_norm, cmap='viridis', s=1.5, alpha=0.7)
            plt.colorbar(scatter1, ax=ax1, label='Intensity', shrink=0.7)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_xlabel("X (meters)", fontsize=11)
        ax1.set_ylabel("Y (meters)", fontsize=11)
        ax1.set_title(f"Raw Point Cloud\n{len(x):,} points", fontsize=12)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Right panel: ST-DBSCAN clustered with tracked IDs
        ax2 = axes[1]
        cluster_legend_items = []

        if len(x) > 0 and len(frame_labels) == len(x):
            noise_mask = frame_labels == -1
            cluster_mask = ~noise_mask

            # Plot noise points in gray
            if noise_mask.any():
                ax2.scatter(x[noise_mask], y[noise_mask], c='lightgray', s=1, alpha=0.3)
                noise_count = noise_mask.sum()
            else:
                noise_count = 0

            # Plot each cluster with tracked color and label
            local_cluster_ids = np.unique(frame_labels[frame_labels >= 0])
            for local_id in local_cluster_ids:
                cmask = frame_labels == local_id
                cx, cy = x[cmask].mean(), y[cmask].mean()

                # Get tracked global ID
                global_id = global_id_map.get((frame_id, local_id), local_id)
                color = tracked_colors[global_id % len(tracked_colors)]

                # Plot cluster points
                ax2.scatter(x[cmask], y[cmask], c=[color], s=2, alpha=0.8)

                # Add cluster ID label at centroid
                ax2.annotate(f'{global_id}', (cx, cy), fontsize=9, fontweight='bold',
                            ha='center', va='center',
                            bbox=dict(boxstyle='circle,pad=0.3', facecolor=color,
                                     edgecolor='black', linewidth=0.5, alpha=0.9),
                            color='white' if sum(color) < 1.5 else 'black')

                cluster_legend_items.append((global_id, color, cmask.sum()))

            num_clusters = len(local_cluster_ids)
            signal_count = cluster_mask.sum()
        else:
            num_clusters = 0
            noise_count = 0
            signal_count = len(x)

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_xlabel("X (meters)", fontsize=11)
        ax2.set_ylabel("Y (meters)", fontsize=11)
        ax2.set_title(f"ST-DBSCAN Clustered\n{num_clusters} clusters, {noise_count:,} noise points", fontsize=12)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # Add legend/key outside the plot on the right
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='gray', label='Noise (filtered)'),
        ]
        # Add top clusters to legend (sorted by point count)
        cluster_legend_items.sort(key=lambda x: -x[2])  # Sort by point count descending
        for global_id, color, count in cluster_legend_items[:10]:  # Show top 10
            legend_elements.append(
                Patch(facecolor=color, edgecolor='black', linewidth=0.5,
                      label=f'Cluster {global_id} ({count:,} pts)')
            )

        # Place legend outside plot on the right
        ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, framealpha=0.95, title='Legend', title_fontsize=10,
                  borderaxespad=0)

        # Main title with frame number
        fig.suptitle(f"Frame {frame_id + 1} of {num_frames}", fontsize=14, fontweight='bold', y=0.98)

        # Adjust layout to make room for legend on right
        plt.tight_layout(rect=[0, 0, 0.88, 0.95])

        # Save frame
        frame_path = temp_dir / f"frame_{idx:04d}.png"
        plt.savefig(frame_path, dpi=150, facecolor='white', edgecolor='none')
        plt.close()
        frame_paths.append(frame_path)

        if (idx + 1) % 5 == 0 or idx == num_frames - 1:
            print(f"    Rendered {idx + 1}/{num_frames} frames...")

    # Combine frames into GIF
    print("  Combining frames into GIF...")
    images = [Image.open(fp) for fp in frame_paths]

    # Save GIF
    gif_path = output_dir / "stdbscan_comparison.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),  # milliseconds per frame
        loop=0  # infinite loop
    )

    # Clean up temp frames
    for fp in frame_paths:
        fp.unlink()
    temp_dir.rmdir()

    print(f"Saved: stdbscan_comparison.gif ({num_frames} frames, {total_tracked} tracked clusters)")


# =============================================================================
# PLY Output Functions
# =============================================================================

def write_ply(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              labels: np.ndarray = None, use_binary: bool = True) -> None:
    """Write point cloud to PLY file with colors based on cluster labels.

    Uses binary format by default for ~10x faster writing.
    """
    num_points = len(x)
    if num_points == 0:
        print(f"Skipping empty point cloud: {path.name}")
        return

    # Generate colors efficiently using vectorized operations
    if labels is not None:
        colors = np.full((num_points, 3), 128, dtype=np.uint8)  # Default gray for noise

        # Vectorized color assignment for clusters
        cluster_mask = labels >= 0
        if cluster_mask.any():
            # Create color lookup table
            cmap = plt.cm.get_cmap('tab20')
            cluster_labels = labels[cluster_mask]
            color_indices = cluster_labels % 20
            # Vectorized colormap lookup
            lut = (np.array([cmap(i)[:3] for i in range(20)]) * 255).astype(np.uint8)
            colors[cluster_mask] = lut[color_indices]
    else:
        # Vectorized intensity coloring
        z_norm = np.clip(z / 255.0, 0, 1)
        colors = (plt.cm.viridis(z_norm)[:, :3] * 255).astype(np.uint8)

    if use_binary:
        # Binary PLY - much faster for large point clouds
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        # Create structured array for efficient binary write
        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
        ])
        data = np.empty(num_points, dtype=dtype)
        data['x'] = x.astype(np.float32)
        data['y'] = y.astype(np.float32)
        data['z'] = z.astype(np.float32)
        data['r'] = colors[:, 0]
        data['g'] = colors[:, 1]
        data['b'] = colors[:, 2]

        with path.open("wb") as fh:
            fh.write(header.encode('ascii'))
            data.tofile(fh)
    else:
        # ASCII PLY - more compatible but slower
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        # Bulk write using numpy - much faster than line-by-line
        data = np.column_stack([
            x.astype(np.float32),
            y.astype(np.float32),
            z.astype(np.float32),
            colors
        ])

        with path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            np.savetxt(fh, data, fmt='%.4f %.4f %.4f %d %d %d')

    print(f"Wrote {num_points:,} points to {path.name}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(data_dir: Path, output_dir: Path,
                 eps_space: float, eps_time: float, min_samples: int,
                 min_frames: int, max_frames: int, no_viz: bool,
                 skip_gif: bool = True, parallel: bool = True,
                 low_memory: bool = False) -> None:
    """Run the complete ST-DBSCAN denoising pipeline.

    Args:
        eps_space: Spatial clustering radius (meters)
        eps_time: Temporal clustering window (frames)
        min_samples: Minimum points to form a cluster
        min_frames: Minimum number of frames a cluster must span (filters single-frame noise)
        max_frames: Maximum frames to process (0 = all)
        no_viz: Skip visualization generation
        parallel: Use multiprocessing for faster CSV loading
        low_memory: Use memory-efficient mode (slower but uses less RAM)
    """

    print("=" * 60)
    print("ST-DBSCAN RADAR POINT CLOUD DENOISING PIPELINE")
    print("=" * 60)

    # Step 1: Discover files
    print("\n[1/5] Discovering data files...")
    gain_files = discover_files(data_dir)
    if not gain_files:
        raise FileNotFoundError(f"No gain folders found in {data_dir}")

    for gain, files in sorted(gain_files.items()):
        print(f"  Gain {gain}: {len(files)} files")

    # Step 2: Group into temporal frames
    print("\n[2/5] Grouping files into temporal frames...")
    frames = group_into_frames(gain_files)
    print(f"  Found {len(frames)} frames")

    if max_frames > 0:
        frames = frames[:max_frames]
        print(f"  Processing first {len(frames)} frames")

    # Step 3: Build point clouds
    print("\n[3/5] Converting radar data to Cartesian point clouds...")

    if parallel and len(frames) > 4:
        print(f"  Using parallel loading with {MAX_WORKERS} workers...")
        frames_data = load_frames_parallel(frames, MAX_WORKERS)
    else:
        # Sequential loading (simpler, uses less memory)
        frames_data = []
        for frame_idx, frame in enumerate(frames):
            fx, fy, fz = load_frame(frame)
            frames_data.append({'x': fx, 'y': fy, 'z': fz})
            if (frame_idx + 1) % 10 == 0:
                print(f"  Processed {frame_idx + 1}/{len(frames)} frames...")

    # Build combined arrays efficiently
    # Pre-calculate total size to avoid repeated allocations
    total_points = sum(len(fd['x']) for fd in frames_data if fd['x'] is not None and len(fd['x']) > 0)
    print(f"  Total points: {total_points:,}")

    if total_points == 0:
        print("  No points found! Check data directory.")
        return

    # Pre-allocate arrays (faster than concatenating lists)
    all_x = np.empty(total_points, dtype=np.float32)
    all_y = np.empty(total_points, dtype=np.float32)
    all_z = np.empty(total_points, dtype=np.float32)
    all_times = np.empty(total_points, dtype=np.float32)

    offset = 0
    for frame_idx, fd in enumerate(frames_data):
        n = len(fd['x'])
        if n > 0:
            all_x[offset:offset+n] = fd['x']
            all_y[offset:offset+n] = fd['y']
            all_z[offset:offset+n] = fd['z']
            all_times[offset:offset+n] = frame_idx
            offset += n

    # Free memory from frames_data if in low memory mode
    if low_memory:
        del frames_data
        gc.collect()
        frames_data = None  # Will reload for visualization if needed

    print(f"  Total points: {len(all_x):,}")

    # Step 4: Apply ST-DBSCAN
    print("\n[4/5] Applying ST-DBSCAN clustering for denoising...")
    print(f"  Parameters: eps_space={eps_space}, eps_time={eps_time}, min_samples={min_samples}, min_frames={min_frames}")

    coords = np.column_stack([all_x, all_y])
    labels = st_dbscan(coords, all_times, eps_space, eps_time, min_samples, min_frames)

    # Calculate statistics
    noise_mask = labels == -1
    signal_mask = ~noise_mask
    num_clusters = len(np.unique(labels[signal_mask]))

    stats = {
        'total_points': len(all_x),
        'noise_points': noise_mask.sum(),
        'signal_points': signal_mask.sum(),
        'num_clusters': num_clusters,
        'noise_reduction_pct': 100.0 * noise_mask.sum() / len(all_x)
    }

    print(f"\n  Results:")
    print(f"    Total points:      {stats['total_points']:,}")
    print(f"    Noise (removed):   {stats['noise_points']:,} ({stats['noise_reduction_pct']:.1f}%)")
    print(f"    Signal (kept):     {stats['signal_points']:,}")
    print(f"    Clusters found:    {stats['num_clusters']}")

    # Step 5: Save outputs
    print("\n[5/5] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save denoised point cloud (signal only)
    denoised_x = all_x[signal_mask]
    denoised_y = all_y[signal_mask]
    denoised_z = all_z[signal_mask]
    denoised_labels = labels[signal_mask]

    write_ply(output_dir / "denoised_point_cloud.ply",
              denoised_x, denoised_y, denoised_z, denoised_labels)

    # Save raw point cloud for comparison
    write_ply(output_dir / "raw_point_cloud.ply", all_x, all_y, all_z)

    # Save statistics CSV
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / "denoising_stats.csv", index=False)
    print(f"Saved: denoising_stats.csv")

    # Save cluster info (vectorized for speed)
    if num_clusters > 0:
        # Use pandas groupby for efficient cluster statistics
        cluster_df = pd.DataFrame({
            'cluster_id': labels[signal_mask],
            'x': denoised_x,
            'y': denoised_y,
            'intensity': denoised_z
        })
        cluster_stats = cluster_df.groupby('cluster_id').agg(
            num_points=('x', 'count'),
            centroid_x=('x', 'mean'),
            centroid_y=('y', 'mean'),
            mean_intensity=('intensity', 'mean')
        ).reset_index()
        cluster_stats.to_csv(output_dir / "clusters.csv", index=False)
        print(f"Saved: clusters.csv")

    # Generate visualizations
    if not no_viz:
        print("\nGenerating visualizations...")

        frame_indices = all_times.astype(int)
        # Reload frames_data if needed for visualization
        if frames_data is None:
            print("  Reloading frame data for visualization...")
            frames_data = load_frames_parallel(frames, MAX_WORKERS) if parallel else [
                load_frame(f) for f in frames
            ]
            frames_data = [{'x': fd[0], 'y': fd[1], 'z': fd[2]} if isinstance(fd, tuple) else fd
                          for fd in frames_data]

        # Static comparison image
        plot_before_after(output_dir, all_x, all_y, all_z,
                         denoised_x, denoised_y, denoised_z, denoised_labels)

        # Temporal cluster visualization
        plot_temporal_clusters(output_dir, frames_data, labels, frame_indices)

        # Noise reduction statistics
        plot_noise_reduction_stats(output_dir, stats)

        # Animated GIF showing frame-by-frame comparison (slow, skip by default)
        if not skip_gif:
            create_comparison_gif(output_dir, frames_data, labels, frame_indices, fps=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def quick_run():
    """Quick run mode - just click run to see 5 frames of clustered data."""
    script_dir = Path(__file__).resolve().parent

    # Try to find data directory
    possible_data_dirs = [
        script_dir.parent / "(.125NM)data_pattern3(.125NM)",
        script_dir / "data",
        script_dir.parent / "data",
    ]

    data_dir = None
    for d in possible_data_dirs:
        if d.exists():
            data_dir = d
            break

    if data_dir is None:
        print("=" * 60)
        print("QUICK RUN MODE")
        print("=" * 60)
        print("\nCould not find data directory. Looked in:")
        for d in possible_data_dirs:
            print(f"  - {d}")
        print("\nPlease either:")
        print("  1. Create one of these directories with gain_XX subfolders")
        print("  2. Run with --data-dir argument:")
        print("     python stdbscan_denoising_pipeline.py --data-dir /path/to/data")
        return

    output_dir = script_dir / "denoising_results"

    print("=" * 60)
    print("QUICK RUN MODE - Processing 5 frames")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("\nThis will generate cluster visualizations automatically.")
    print("For more options, run: python stdbscan_denoising_pipeline.py --help\n")

    run_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        eps_space=DEFAULT_EPS_SPACE,
        eps_time=DEFAULT_EPS_TIME,
        min_samples=DEFAULT_MIN_SAMPLES,
        min_frames=DEFAULT_MIN_FRAMES,  # Require clusters to span multiple frames
        max_frames=5,  # Just 5 frames for quick visualization
        no_viz=False,  # Generate visualizations
        parallel=False,  # Simpler, less memory
        low_memory=True  # Be gentle on RAM
    )

    # Open output folder when done (Windows)
    import subprocess
    import sys
    if sys.platform == 'win32' and output_dir.exists():
        subprocess.run(['explorer', str(output_dir)], check=False)
    elif output_dir.exists():
        print(f"\nResults saved to: {output_dir}")


def main():
    import sys

    # If no arguments provided, run in quick mode
    if len(sys.argv) == 1:
        quick_run()
        return

    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "(.125NM)data_pattern3(.125NM)"
    default_output_dir = script_dir / "denoising_results"

    parser = argparse.ArgumentParser(
        description="ST-DBSCAN Radar Point Cloud Denoising Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stdbscan_denoising_pipeline.py                # Quick run (5 frames)
  python stdbscan_denoising_pipeline.py --max-frames 50
  python stdbscan_denoising_pipeline.py --eps-space 10.0 --min-samples 20
  python stdbscan_denoising_pipeline.py --no-viz      # Skip visualizations
  python stdbscan_denoising_pipeline.py --low-memory  # Reduce RAM usage
        """
    )

    parser.add_argument("--data-dir", type=Path, default=default_data_dir,
                       help="Directory containing gain_XX folders")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir,
                       help="Output directory for results")
    parser.add_argument("--eps-space", type=float, default=DEFAULT_EPS_SPACE,
                       help=f"Spatial clustering radius in meters (default: {DEFAULT_EPS_SPACE})")
    parser.add_argument("--eps-time", type=float, default=DEFAULT_EPS_TIME,
                       help=f"Temporal clustering window in frames (default: {DEFAULT_EPS_TIME})")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                       help=f"Minimum points to form a cluster (default: {DEFAULT_MIN_SAMPLES})")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES,
                       help=f"Minimum frames a cluster must span to be valid (default: {DEFAULT_MIN_FRAMES})")
    parser.add_argument("--max-frames", type=int, default=5,
                       help="Maximum frames to process (default: 5, 0 = all)")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization generation")
    parser.add_argument("--skip-gif", action="store_true",
                       help="Skip GIF generation (much faster, still creates PNGs)")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel CSV loading (uses less memory)")
    parser.add_argument("--low-memory", action="store_true",
                       help="Memory-efficient mode (frees intermediate data)")

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        eps_space=args.eps_space,
        eps_time=args.eps_time,
        min_samples=args.min_samples,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        no_viz=args.no_viz,
        skip_gif=args.skip_gif,
        parallel=not args.no_parallel,
        low_memory=args.low_memory
    )


if __name__ == "__main__":
    main()

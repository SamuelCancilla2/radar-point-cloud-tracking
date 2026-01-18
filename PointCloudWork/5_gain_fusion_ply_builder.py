#!/usr/bin/env python3
"""
Gain Fusion Point Cloud Builder

Takes 3 separate gain values (40, 50, 70/75) and fuses them into point clouds:
- Each frame fuses all gains together
- Intensity is shown as absolute value
- All gains start from the same origin point (no Z offset)
- Sequential frames can be stacked or viewed separately

This creates PLY files suitable for visualization in tools like CloudCompare, MeshLab, etc.

Usage:
    python 5_gain_fusion_ply_builder.py --data-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# =============================================================================
# Configuration
# =============================================================================

SUPPORTED_GAINS = {40, 50, 70, 75}

# Colors for each gain level (for visualization)
GAIN_COLORS = {
    40: (0, 114, 255),    # Blue - lowest gain, highest sensitivity
    50: (0, 200, 83),     # Green - medium gain
    70: (255, 165, 0),    # Orange
    75: (255, 87, 34),    # Orange-red - highest gain, lowest sensitivity
}

# Radar parameters
ANGLE_SCALE = 360.0 / 8196.0
NUM_ECHO_COLUMNS = 1024

# Processing parameters
INTENSITY_THRESHOLD = 5.0     # Minimum intensity to keep
POINT_STRIDE = 8              # Keep every Nth point
MAX_TIME_DIFF_MS = 2000       # Max time diff to be same frame

# Intensity normalization
NORMALIZE_INTENSITY = True    # Normalize to 0-255 range
INTENSITY_PERCENTILE = 99     # Use this percentile as max for normalization


# =============================================================================
# File Parsing
# =============================================================================

def parse_timestamp(filename: str) -> Tuple[datetime, int]:
    """Parse timestamp from filename like '20250813_142602_181.csv'."""
    match = re.match(r"(\d{8})_(\d{6})_(\d{3})\.csv", filename)
    if not match:
        raise ValueError(f"Cannot parse timestamp from {filename}")

    date_str, time_str, ms_str = match.groups()
    dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    ms = int(ms_str)
    total_ms = int(dt.timestamp() * 1000) + ms

    return dt, total_ms


def load_radar_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load radar CSV and convert to Cartesian. Returns (x, y, intensity, gain)."""
    col_names = ["Status", "Scale", "Range", "Gain", "Angle"] + [f"Echo_{i}" for i in range(NUM_ECHO_COLUMNS)]

    try:
        df = pd.read_csv(path, header=None, names=col_names, skiprows=1, engine="c")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.array([]), np.array([]), np.array([]), 0

    if df.empty:
        return np.array([]), np.array([]), np.array([]), 0

    gain = int(df["Gain"].iloc[0])
    angles_rad = np.deg2rad(df["Angle"].to_numpy(np.float32) * ANGLE_SCALE)
    echo_data = df.iloc[:, 5:].fillna(0).to_numpy(np.float32)
    max_ranges = df["Scale"].to_numpy(np.float32)
    num_bins = echo_data.shape[1]

    range_res = max_ranges[:, None] / num_bins
    ranges = range_res * np.arange(num_bins, dtype=np.float32)

    x = ranges * np.cos(angles_rad[:, None])
    y = ranges * np.sin(angles_rad[:, None])

    # Apply threshold and flatten
    mask = echo_data > INTENSITY_THRESHOLD
    x_flat = x[mask]
    y_flat = y[mask]
    intensity_flat = echo_data[mask]

    # Apply stride
    if POINT_STRIDE > 1:
        x_flat = x_flat[::POINT_STRIDE]
        y_flat = y_flat[::POINT_STRIDE]
        intensity_flat = intensity_flat[::POINT_STRIDE]

    return x_flat, y_flat, intensity_flat, gain


def discover_files(data_dir: Path) -> Dict[int, List[Path]]:
    """Discover all CSV files organized by gain."""
    files_by_gain: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)

    for gain_dir in data_dir.iterdir():
        if not gain_dir.is_dir():
            continue

        match = re.search(r"gain[_-]?(\d+)", gain_dir.name, re.IGNORECASE)
        if not match:
            continue

        gain = int(match.group(1))
        if gain not in SUPPORTED_GAINS:
            continue

        for csv_path in gain_dir.glob("*.csv"):
            try:
                _, ts_ms = parse_timestamp(csv_path.name)
                files_by_gain[gain].append((ts_ms, csv_path))
            except ValueError:
                continue

    result = {}
    for gain, files in files_by_gain.items():
        files.sort(key=lambda x: x[0])
        result[gain] = [f[1] for f in files]

    return result


def group_files_by_frame(files_by_gain: Dict[int, List[Path]]) -> List[Dict[int, Path]]:
    """Group files across gains into frames based on timestamp proximity."""
    all_files: List[Tuple[int, int, Path]] = []

    for gain, paths in files_by_gain.items():
        for path in paths:
            _, ts_ms = parse_timestamp(path.name)
            all_files.append((ts_ms, gain, path))

    all_files.sort(key=lambda x: x[0])

    frames: List[Dict[int, Path]] = []
    current_frame: Dict[int, Path] = {}
    frame_start_ts = None

    for ts_ms, gain, path in all_files:
        if frame_start_ts is None:
            frame_start_ts = ts_ms
            current_frame = {gain: path}
        elif ts_ms - frame_start_ts <= MAX_TIME_DIFF_MS:
            if gain not in current_frame:
                current_frame[gain] = path
        else:
            if current_frame:
                frames.append(current_frame)
            frame_start_ts = ts_ms
            current_frame = {gain: path}

    if current_frame:
        frames.append(current_frame)

    return frames


# =============================================================================
# Point Cloud Building
# =============================================================================

def fuse_gains_absolute(frame_files: Dict[int, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse multiple gains into a single point cloud with absolute intensity.

    Returns (x, y, intensity, gain_labels).
    """
    all_x, all_y, all_intensity, all_gains = [], [], [], []

    for gain, path in sorted(frame_files.items()):
        x, y, intensity, _ = load_radar_csv(path)
        if len(x) == 0:
            continue

        all_x.append(x)
        all_y.append(y)
        all_intensity.append(intensity)
        all_gains.append(np.full(len(x), gain, dtype=np.int32))

    if not all_x:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_intensity),
        np.concatenate(all_gains)
    )


def fuse_gains_max(frame_files: Dict[int, Path], grid_resolution: float = 1.0
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse multiple gains by taking maximum intensity at each spatial location.

    This creates a cleaner point cloud where overlapping gains contribute
    their maximum intensity value.

    Returns (x, y, max_intensity).
    """
    all_x, all_y, all_intensity = [], [], []

    for gain, path in sorted(frame_files.items()):
        x, y, intensity, _ = load_radar_csv(path)
        if len(x) == 0:
            continue

        all_x.append(x)
        all_y.append(y)
        all_intensity.append(intensity)

    if not all_x:
        return np.array([]), np.array([]), np.array([])

    x_combined = np.concatenate(all_x)
    y_combined = np.concatenate(all_y)
    intensity_combined = np.concatenate(all_intensity)

    # Grid-based max pooling
    x_min, x_max = x_combined.min(), x_combined.max()
    y_min, y_max = y_combined.min(), y_combined.max()

    x_bins = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
    y_bins = int(np.ceil((y_max - y_min) / grid_resolution)) + 1

    # Digitize points to grid cells
    x_idx = ((x_combined - x_min) / grid_resolution).astype(int)
    y_idx = ((y_combined - y_min) / grid_resolution).astype(int)

    # Max intensity per cell
    max_grid = np.zeros((x_bins, y_bins), dtype=np.float32)
    np.maximum.at(max_grid, (x_idx, y_idx), intensity_combined)

    # Extract non-zero cells
    valid = max_grid > 0
    valid_y, valid_x = np.where(valid.T)  # Note: transposed for correct indexing

    out_x = x_min + valid_x * grid_resolution + grid_resolution / 2
    out_y = y_min + valid_y * grid_resolution + grid_resolution / 2
    out_intensity = max_grid[valid_x, valid_y]

    return out_x, out_y, out_intensity


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    """Normalize intensity to 0-255 range."""
    if len(intensity) == 0:
        return intensity

    # Use percentile to handle outliers
    max_val = np.percentile(intensity, INTENSITY_PERCENTILE)
    min_val = np.min(intensity)

    if max_val <= min_val:
        return np.zeros_like(intensity)

    normalized = (intensity - min_val) / (max_val - min_val) * 255.0
    return np.clip(normalized, 0, 255)


def intensity_to_rgb(intensity: np.ndarray) -> np.ndarray:
    """Convert intensity values to RGB colors using a colormap."""
    # Use a heat-like colormap: blue (low) -> green -> yellow -> red (high)
    normalized = intensity / 255.0

    rgb = np.zeros((len(intensity), 3), dtype=np.uint8)

    # Blue to cyan
    mask1 = normalized < 0.25
    t = normalized[mask1] * 4
    rgb[mask1, 0] = 0
    rgb[mask1, 1] = (t * 255).astype(np.uint8)
    rgb[mask1, 2] = 255

    # Cyan to green
    mask2 = (normalized >= 0.25) & (normalized < 0.5)
    t = (normalized[mask2] - 0.25) * 4
    rgb[mask2, 0] = 0
    rgb[mask2, 1] = 255
    rgb[mask2, 2] = ((1 - t) * 255).astype(np.uint8)

    # Green to yellow
    mask3 = (normalized >= 0.5) & (normalized < 0.75)
    t = (normalized[mask3] - 0.5) * 4
    rgb[mask3, 0] = (t * 255).astype(np.uint8)
    rgb[mask3, 1] = 255
    rgb[mask3, 2] = 0

    # Yellow to red
    mask4 = normalized >= 0.75
    t = (normalized[mask4] - 0.75) * 4
    rgb[mask4, 0] = 255
    rgb[mask4, 1] = ((1 - t) * 255).astype(np.uint8)
    rgb[mask4, 2] = 0

    return rgb


def gain_to_rgb(gains: np.ndarray) -> np.ndarray:
    """Convert gain labels to RGB colors."""
    rgb = np.zeros((len(gains), 3), dtype=np.uint8)

    for gain, color in GAIN_COLORS.items():
        mask = gains == gain
        rgb[mask] = color

    return rgb


# =============================================================================
# PLY Writing
# =============================================================================

def write_ply(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              colors: np.ndarray) -> None:
    """Write ASCII PLY file with RGB colors."""
    num_points = len(x)

    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        for xp, yp, zp, (r, g, b) in zip(x, y, z, colors):
            fh.write(f"{xp:.4f} {yp:.4f} {zp:.4f} {r} {g} {b}\n")

    print(f"  Wrote {num_points:,} points to {path.name}")


def write_ply_fast(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   colors: np.ndarray) -> None:
    """Write PLY file using numpy for speed."""
    num_points = len(x)

    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with path.open("w", encoding="utf-8") as fh:
        fh.write(header)

    # Prepare data array
    data = np.column_stack([
        x.astype(np.float32),
        y.astype(np.float32),
        z.astype(np.float32),
        colors[:, 0].astype(int),
        colors[:, 1].astype(int),
        colors[:, 2].astype(int)
    ])

    with path.open("a", encoding="utf-8") as fh:
        np.savetxt(fh, data, fmt="%.4f %.4f %.4f %d %d %d")

    print(f"  Wrote {num_points:,} points to {path.name}")


# =============================================================================
# Visualization
# =============================================================================

def plot_point_cloud(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     colors: np.ndarray, output_path: Path, title: str) -> None:
    """Create a 2D bird's-eye view of the point cloud."""
    if not HAS_MPL:
        return

    # Subsample for plotting
    max_plot_points = 500000
    if len(x) > max_plot_points:
        idx = np.random.choice(len(x), max_plot_points, replace=False)
        x, y, z, colors = x[idx], y[idx], z[idx], colors[idx]

    fig, ax = plt.subplots(figsize=(12, 12))

    # Normalize colors to 0-1
    c = colors.astype(np.float32) / 255.0

    scatter = ax.scatter(x, y, c=c, s=0.5, alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_3d_point_cloud(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        colors: np.ndarray, output_path: Path, title: str) -> None:
    """Create a 3D visualization of the point cloud."""
    if not HAS_MPL:
        return

    # Subsample for plotting
    max_plot_points = 200000
    if len(x) > max_plot_points:
        idx = np.random.choice(len(x), max_plot_points, replace=False)
        x, y, z, colors = x[idx], y[idx], z[idx], colors[idx]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    c = colors.astype(np.float32) / 255.0

    ax.scatter(x, y, z, c=c, s=0.5, alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Intensity')
    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Main Pipeline
# =============================================================================

def build_individual_frames(data_dir: Path, output_dir: Path, max_frames: int = 0,
                           mode: str = "absolute") -> None:
    """Build individual PLY files for each frame."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering files...")
    files_by_gain = discover_files(data_dir)

    if not files_by_gain:
        print("No data files found!")
        return

    for gain, files in files_by_gain.items():
        print(f"  Gain {gain}: {len(files)} files")

    print("\nGrouping into frames...")
    frame_files = group_files_by_frame(files_by_gain)
    print(f"  Found {len(frame_files)} frames")

    if max_frames > 0:
        frame_files = frame_files[:max_frames]
        print(f"  Processing first {len(frame_files)} frames")

    print(f"\nBuilding point clouds (mode={mode})...")

    for i, ff in enumerate(frame_files):
        gains_in_frame = sorted(ff.keys())
        gain_str = "_".join(str(g) for g in gains_in_frame)

        if mode == "max":
            x, y, intensity = fuse_gains_max(ff)
        else:  # absolute
            x, y, intensity, _ = fuse_gains_absolute(ff)

        if len(x) == 0:
            continue

        # Z coordinate is intensity (absolute value)
        if NORMALIZE_INTENSITY:
            z = normalize_intensity(intensity)
        else:
            z = intensity

        # Color by intensity
        colors = intensity_to_rgb(z)

        # Save PLY
        ply_path = output_dir / f"frame_{i:04d}_gains_{gain_str}.ply"
        write_ply_fast(ply_path, x, y, z, colors)

        # Save preview PNG every 10 frames
        if i % 10 == 0:
            png_path = output_dir / f"frame_{i:04d}_preview.png"
            plot_point_cloud(x, y, z, colors, png_path,
                           f"Frame {i} (Gains: {gain_str})")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(frame_files)} frames")

    print(f"\nDone! PLY files saved to {output_dir}")


def build_stacked_sequence(data_dir: Path, output_dir: Path, max_frames: int = 100,
                           time_spacing: float = 10.0, mode: str = "absolute") -> None:
    """
    Build a single PLY with sequential frames stacked in the Z direction.

    This allows viewing the temporal evolution in 3D.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering files...")
    files_by_gain = discover_files(data_dir)

    if not files_by_gain:
        print("No data files found!")
        return

    print("\nGrouping into frames...")
    frame_files = group_files_by_frame(files_by_gain)

    if max_frames > 0:
        frame_files = frame_files[:max_frames]

    print(f"  Processing {len(frame_files)} frames")

    all_x, all_y, all_z, all_colors = [], [], [], []

    for i, ff in enumerate(frame_files):
        if mode == "max":
            x, y, intensity = fuse_gains_max(ff)
        else:
            x, y, intensity, _ = fuse_gains_absolute(ff)

        if len(x) == 0:
            continue

        if NORMALIZE_INTENSITY:
            intensity_norm = normalize_intensity(intensity)
        else:
            intensity_norm = intensity

        # Z = time layer (frame index * spacing)
        z = np.full_like(x, i * time_spacing)

        # Color by intensity
        colors = intensity_to_rgb(intensity_norm)

        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
        all_colors.append(colors)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(frame_files)} frames")

    if not all_x:
        print("No points generated!")
        return

    # Combine all frames
    x_combined = np.concatenate(all_x)
    y_combined = np.concatenate(all_y)
    z_combined = np.concatenate(all_z)
    colors_combined = np.concatenate(all_colors)

    print(f"\nTotal points: {len(x_combined):,}")

    # Save stacked PLY
    ply_path = output_dir / f"temporal_stack_{len(frame_files)}frames.ply"
    write_ply_fast(ply_path, x_combined, y_combined, z_combined, colors_combined)

    # Save preview
    png_path = output_dir / "temporal_stack_preview.png"
    plot_3d_point_cloud(x_combined, y_combined, z_combined, colors_combined,
                       png_path, f"Temporal Stack ({len(frame_files)} frames)")

    print(f"Done! Stacked PLY saved to {ply_path}")


def build_gain_comparison(data_dir: Path, output_dir: Path, frame_idx: int = 0) -> None:
    """
    Build separate PLY files for each gain at a single frame for comparison.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering files...")
    files_by_gain = discover_files(data_dir)

    if not files_by_gain:
        print("No data files found!")
        return

    print("\nGrouping into frames...")
    frame_files = group_files_by_frame(files_by_gain)

    if frame_idx >= len(frame_files):
        print(f"Frame {frame_idx} not found. Only {len(frame_files)} frames available.")
        return

    ff = frame_files[frame_idx]
    print(f"\nProcessing frame {frame_idx}...")

    for gain, path in sorted(ff.items()):
        x, y, intensity, _ = load_radar_csv(path)

        if len(x) == 0:
            print(f"  Gain {gain}: No points")
            continue

        if NORMALIZE_INTENSITY:
            z = normalize_intensity(intensity)
        else:
            z = intensity

        # Color by gain
        colors = np.full((len(x), 3), GAIN_COLORS[gain], dtype=np.uint8)

        ply_path = output_dir / f"frame_{frame_idx:04d}_gain_{gain}.ply"
        write_ply_fast(ply_path, x, y, z, colors)

        png_path = output_dir / f"frame_{frame_idx:04d}_gain_{gain}.png"
        plot_point_cloud(x, y, z, colors, png_path, f"Frame {frame_idx} - Gain {gain}")

        print(f"  Gain {gain}: {len(x):,} points")

    # Also build fused version
    x, y, intensity, gains = fuse_gains_absolute(ff)
    if len(x) > 0:
        if NORMALIZE_INTENSITY:
            z = normalize_intensity(intensity)
        else:
            z = intensity

        colors = gain_to_rgb(gains)

        ply_path = output_dir / f"frame_{frame_idx:04d}_fused_by_gain.ply"
        write_ply_fast(ply_path, x, y, z, colors)

        # Also with intensity coloring
        colors_intensity = intensity_to_rgb(z)
        ply_path = output_dir / f"frame_{frame_idx:04d}_fused_by_intensity.ply"
        write_ply_fast(ply_path, x, y, z, colors_intensity)

    print(f"\nDone! Files saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build gain-fused PLY point clouds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  individual  - Create separate PLY for each frame
  stacked     - Create single PLY with all frames stacked in Z
  comparison  - Create separate PLYs for each gain at one frame

Examples:
  python 5_gain_fusion_ply_builder.py individual --data-dir ./data --output-dir ./ply
  python 5_gain_fusion_ply_builder.py stacked --data-dir ./data --output-dir ./ply --max-frames 50
  python 5_gain_fusion_ply_builder.py comparison --data-dir ./data --output-dir ./ply --frame 0
        """
    )

    script_dir = Path(__file__).resolve().parent
    default_data = script_dir.parent / "(.125NM)data_pattern3(.125NM)"
    default_output = script_dir / "gain_fused_ply"

    subparsers = parser.add_subparsers(dest="command", help="Build mode")

    # Individual frames
    p_ind = subparsers.add_parser("individual", help="Build individual frame PLYs")
    p_ind.add_argument("--data-dir", type=Path, default=default_data)
    p_ind.add_argument("--output-dir", type=Path, default=default_output / "individual")
    p_ind.add_argument("--max-frames", type=int, default=0)
    p_ind.add_argument("--mode", choices=["absolute", "max"], default="absolute",
                      help="Fusion mode: absolute (keep all), max (max per cell)")

    # Stacked sequence
    p_stack = subparsers.add_parser("stacked", help="Build stacked temporal PLY")
    p_stack.add_argument("--data-dir", type=Path, default=default_data)
    p_stack.add_argument("--output-dir", type=Path, default=default_output / "stacked")
    p_stack.add_argument("--max-frames", type=int, default=100)
    p_stack.add_argument("--time-spacing", type=float, default=10.0,
                        help="Z spacing between frames")
    p_stack.add_argument("--mode", choices=["absolute", "max"], default="absolute")

    # Gain comparison
    p_comp = subparsers.add_parser("comparison", help="Build per-gain PLYs for one frame")
    p_comp.add_argument("--data-dir", type=Path, default=default_data)
    p_comp.add_argument("--output-dir", type=Path, default=default_output / "comparison")
    p_comp.add_argument("--frame", type=int, default=0, help="Frame index to analyze")

    args = parser.parse_args()

    if args.command == "individual":
        build_individual_frames(args.data_dir, args.output_dir, args.max_frames, args.mode)
    elif args.command == "stacked":
        build_stacked_sequence(args.data_dir, args.output_dir, args.max_frames,
                              args.time_spacing, args.mode)
    elif args.command == "comparison":
        build_gain_comparison(args.data_dir, args.output_dir, args.frame)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate stacked PLY point clouds (offset + flat) from radar sweeps that have already
been sliced into single rotations per gain.

Outputs
-------
- frame_stack_flat_v3.ply/png : all gains co-located (flat)
- frame_stack_v3.ply/png      : gains vertically offset for visual separation
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ModuleNotFoundError:
    HAS_MPL = False

PROJECT_ROOT = Path(__file__).resolve().parent
SWEEP_DIR = PROJECT_ROOT / "1.5_Folder"

RANGE_BIN_WIDTH_M = 0.5
RANGE_START_M = 0.0
INTENSITY_THRESHOLD = 0.0  # tweak if you want to drop very low echoes
# Keep every Nth point to reduce clutter / space things out.
# These defaults will further auto-scale based on file size.
POINT_STRIDE = 16
TARGET_MAX_POINTS_PER_GAIN = 10_000_000
TARGET_MAX_POINTS_STACK = 20_000_000
PLOT_MAX_POINTS = 1_000_000  # cap plotted points for speed

# RGB colors (0-255) used when stacking clouds so the gains are easy to spot.
GAIN_COLORS = {
    40: (0, 114, 255),   # blue-ish
    50: (0, 200, 83),    # green
    75: (255, 87, 34),   # orange
}

# Vertical offsets applied only in the stacked view so layers are separated visually.
GAIN_Z_OFFSETS = {
    75: 0.0,    # bottom layer
    50: 250.0,  # middle layer
    40: 500.0,  # top layer
}


def load_radar_sweep(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (angles_rad, intensity_matrix) matching the original gain-40 conversion approach."""
    df = pd.read_csv(path, header=None, skiprows=1)
    intensities = df.iloc[:, 5:].to_numpy(dtype=np.float32)
    num_angles = len(df)
    angles_rad = np.linspace(0.0, 2 * np.pi, num_angles,
                             endpoint=False, dtype=np.float32)
    return angles_rad, intensities


def sweep_to_points(
    angles_rad: np.ndarray,
    intensities: np.ndarray,
    min_intensity: float = INTENSITY_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast the sweep into Cartesian coordinates and return filtered/subsampled points."""
    num_bins = intensities.shape[1]
    ranges = RANGE_START_M + \
        np.arange(num_bins, dtype=np.float32) * RANGE_BIN_WIDTH_M
    range_grid = ranges[None, :]

    cos_angles = np.cos(angles_rad)[:, None].astype(np.float32)
    sin_angles = np.sin(angles_rad)[:, None].astype(np.float32)

    x = range_grid * cos_angles
    y = range_grid * sin_angles
    z = intensities

    mask = z > min_intensity
    x_pts, y_pts, z_pts = x[mask], y[mask], z[mask]
    if POINT_STRIDE > 1:
        x_pts = x_pts[::POINT_STRIDE]
        y_pts = y_pts[::POINT_STRIDE]
        z_pts = z_pts[::POINT_STRIDE]
    return x_pts, y_pts, z_pts


def load_points(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load points from either a flat x/y/z CSV (output of 1_CSVtoCartesian.py)
    or from a raw radar sweep grid.
    """
    df_preview = pd.read_csv(path, nrows=2)
    lower_cols = [c.lower() for c in df_preview.columns]
    has_xyz_header = {"x", "y", "z"}.issubset(lower_cols)

    if has_xyz_header or (df_preview.shape[1] == 3 and df_preview.columns[0] != 0):
        df = pd.read_csv(path)
        col_map = {c.lower(): c for c in df.columns}
        x = df[col_map.get("x", df.columns[0])].to_numpy(np.float32)
        y = df[col_map.get("y", df.columns[1])].to_numpy(np.float32)
        z = df[col_map.get("z", df.columns[2])].to_numpy(np.float32)
        return x, y, z

    # Fallback to raw radar sweep interpretation.
    angles_rad, intensities = load_radar_sweep(path)
    return sweep_to_points(angles_rad, intensities)


def apply_stride(x: np.ndarray, y: np.ndarray, z: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strided subsets of the point arrays."""
    if stride <= 1:
        return x, y, z
    return x[::stride], y[::stride], z[::stride]


def intensity_colors(values: np.ndarray) -> np.ndarray:
    """Map scalar intensity values to grayscale RGB."""
    clipped = np.clip(values, 0, 255)
    colors = clipped.astype(np.uint8)
    return np.stack([colors, colors, colors], axis=1)


def gain_colors(values: np.ndarray, gain: int) -> np.ndarray:
    """Return constant RGB for a gain."""
    rgb = np.array(GAIN_COLORS.get(gain, (180, 180, 180)), dtype=np.uint8)
    return np.repeat(rgb[None, :], values.size, axis=0)


def find_sweeps(directory: Path) -> Dict[int, Path]:
    """
    Discover gain-specific sweep CSVs in the target directory.

    Expected filenames contain 'gain_<number>', e.g. '1.5_TEST_gain_75.csv'.
    """
    sweeps: Dict[int, Path] = {}
    for path in sorted(directory.glob("*.csv")):
        match = re.search(r"gain[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
        if not match:
            continue
        gain = int(match.group(1))
        sweeps[gain] = path

    if not sweeps:
        raise FileNotFoundError(f"No gain CSVs found in {directory}")
    return sweeps


def write_ply(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY file with per-point RGB."""
    num_points = x.size
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
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        for xp, yp, zp, (r, g, b) in zip(x, y, z, colors, strict=False):
            fh.write(f"{xp:.6f} {yp:.6f} {zp:.6f} {r} {g} {b}\n")


def plot_cloud(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, title: str) -> None:
    """Save a lightweight 3D scatter plot of the cloud, subsampling for speed."""
    if not HAS_MPL:
        print("matplotlib not installed; skipping preview plot.")
        return
    n_points = x.size
    plot_stride = max(1, int(np.ceil(n_points / PLOT_MAX_POINTS)))
    if plot_stride > 1:
        x = x[::plot_stride]
        y = y[::plot_stride]
        z = z[::plot_stride]
        colors = colors[::plot_stride]
        print(f"plot subsample: {x.size:,} points (plot_stride={plot_stride})")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Colors are uint8; matplotlib expects 0-1 floats.
    scatter_colors = colors.astype(np.float32) / 255.0
    ax.scatter(x, y, z, c=scatter_colors, s=1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Intensity")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_stack_variant(
    name_stem: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    colors: np.ndarray,
    title: str,
) -> None:
    """Write a stacked PLY/PNG pair, applying stack stride for size control."""
    stack_stride = max(1, int(np.ceil(x.size / TARGET_MAX_POINTS_STACK)))
    if stack_stride > 1:
        x, y, z = apply_stride(x, y, z, stack_stride)
        colors = colors[::stack_stride]

    stack_ply = SWEEP_DIR / f"{name_stem}.ply"
    write_ply(stack_ply, x, y, z, colors)
    print(
        f"{name_stem}: {x.size:,} points"
        f" (stack_stride={stack_stride})"
        f" -> {stack_ply.name}"
    )

    plot_path = SWEEP_DIR / f"{name_stem}.png"
    plot_cloud(plot_path, x, y, z, colors, title)


def main() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    sweep_files = find_sweeps(SWEEP_DIR)
    combined_points = []
    combined_colors = []

    for gain, sweep_path in sweep_files.items():
        x, y, z = load_points(sweep_path)
        base_points = x.size

        # Auto-raise stride if this gain file is very large.
        gain_stride = max(POINT_STRIDE, int(
            np.ceil(base_points / TARGET_MAX_POINTS_PER_GAIN)))
        x, y, z = apply_stride(x, y, z, gain_stride)

        combined_points.append((gain, x, y, z))
        combined_colors.append(gain_colors(z, gain))

        print(
            f"gain {gain}: {x.size:,} points"
            f" (stride={gain_stride})"
        )

    # Stack all gains into one cloud with tinted colors.
    base_x = np.concatenate([pts[1] for pts in combined_points])
    base_y = np.concatenate([pts[2] for pts in combined_points])
    base_z = np.concatenate([pts[3] for pts in combined_points])
    all_colors = np.concatenate(combined_colors)

    title_gains = "/".join(str(g) for g in sweep_files.keys())

    # Offset stack (separated layers).
    offset_z = np.concatenate(
        [pts[3] + GAIN_Z_OFFSETS.get(pts[0], 0.0) for pts in combined_points]
    )
    write_stack_variant(
        "frame_stack_v3",
        base_x,
        base_y,
        offset_z,
        all_colors,
        f"Range 2 stacked gains ({title_gains})",
    )

    # Flat stack (all gains start from the bottom / no offsets).
    write_stack_variant(
        "frame_stack_flat_v3",
        base_x,
        base_y,
        base_z,
        all_colors,
        f"Range 2 stacked gains flat ({title_gains})",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate a lightweight PNG preview from an existing ASCII PLY point cloud.

Features
--------
- Takes a PLY path and writes a PNG separately from the main pipeline.
- Subsamples to a configurable maximum number of points for speed.
- Uses small markers with partial transparency to reduce overdraw.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ModuleNotFoundError:
    HAS_MPL = False


def load_ply(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimal ASCII PLY reader for x/y/z plus optional RGB (uchar).
    Returns x, y, z, colors (uint8 RGB; default gray if absent).
    """
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    if not lines or not lines[0].strip().startswith("ply"):
        raise ValueError(f"{path} is not a PLY file")

    num_vertices = None
    header_end = None
    prop_names = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            num_vertices = int(stripped.split()[-1])
        elif stripped.startswith("property"):
            parts = stripped.split()
            prop_names.append(parts[-1])
        elif stripped == "end_header":
            header_end = idx + 1
            break

    if num_vertices is None or header_end is None:
        raise ValueError(f"Could not parse header for {path}")

    data_lines = lines[header_end:header_end + num_vertices]
    if len(data_lines) < num_vertices:
        raise ValueError(f"Expected {num_vertices} vertices, found {len(data_lines)}")

    data = np.fromiter(
        (float(item) for line in data_lines for item in line.split()),
        dtype=np.float32,
        count=len(data_lines[0].split()) * num_vertices,
    ).reshape(num_vertices, -1)

    prop_idx = {name: i for i, name in enumerate(prop_names)}
    try:
        x = data[:, prop_idx["x"]]
        y = data[:, prop_idx["y"]]
        z = data[:, prop_idx["z"]]
    except KeyError as exc:
        raise ValueError(f"PLY missing x/y/z properties: {path}") from exc

    if {"red", "green", "blue"} <= prop_idx.keys():
        r = data[:, prop_idx["red"]]
        g = data[:, prop_idx["green"]]
        b = data[:, prop_idx["blue"]]
        colors = np.stack([r, g, b], axis=1).astype(np.uint8)
    else:
        gray = np.full_like(x, 180, dtype=np.uint8)
        colors = np.stack([gray, gray, gray], axis=1)

    return x, y, z, colors


def subsample(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Randomly subsample to at most max_points; return stride info for logging."""
    n = x.size
    if n <= max_points:
        return x, y, z, colors, 1
    idx = np.random.choice(n, max_points, replace=False)
    return x[idx], y[idx], z[idx], colors[idx], int(np.ceil(n / max_points))


def plot_cloud(out_path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, title: str, alpha: float) -> None:
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for plotting but is not installed.")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter_colors = colors.astype(np.float32) / 255.0
    ax.scatter(x, y, z, c=scatter_colors, s=0.35, alpha=alpha)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z / Intensity")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_ply = script_dir / "1.5_Folder" / "frame_stack_v2.ply"
    default_output = default_ply.with_name(f"{default_ply.stem}_v2_image_distance.png")

    p = argparse.ArgumentParser(description="Generate a PNG preview from a PLY point cloud.")
    p.add_argument("--ply", type=Path, default=default_ply, help=f"Path to ASCII PLY file to preview (default: {default_ply}).")
    p.add_argument("--output", type=Path, default=default_output, help=f"PNG output path (default: {default_output}).")
    p.add_argument("--max-points", type=int, default=1_000_000, help="Maximum points to plot (randomly sampled if larger).")
    p.add_argument("--alpha", type=float, default=0.5, help="Marker opacity (0-1).")
    args = p.parse_args()

    x, y, z, colors = load_ply(args.ply)
    x, y, z, colors, stride = subsample(x, y, z, colors, args.max_points)
    print(f"Loaded {colors.shape[0]:,} points for plotting (approx stride={stride}) -> {args.output.name}")

    plot_cloud(args.output, x, y, z, colors, f"{args.ply.name}", alpha=args.alpha)
    print(f"PNG saved to {args.output}")


if __name__ == "__main__":
    main()

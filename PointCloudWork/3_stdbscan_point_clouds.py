#!/usr/bin/env python3
"""
Run ST-DBSCAN on stacked point clouds (offset and flat) and save labels/preview.
Defaults to 1.5_Folder/frame_stack_v3.ply and frame_stack_flat_v3.ply.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import BallTree

# ST-DBSCAN and subsampling defaults
EPS_SPACE = 5.0      # spatial eps
EPS_TIME = 1.0       # temporal eps (in time-step units)
MIN_SAMPLES = 10
MAX_POINTS = 10_000_000

# Gain color lookup used to infer "time step" when none is provided.
GAIN_COLORS: Dict[int, Tuple[int, int, int]] = {
    40: (0, 114, 255),
    50: (0, 200, 83),
    75: (255, 87, 34),
}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ModuleNotFoundError:
    HAS_MPL = False


def load_ply(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minimal ASCII PLY reader for x/y/z plus optional RGB."""
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    if not lines or not lines[0].strip().startswith("ply"):
        raise ValueError(f"{path} is not a PLY file")

    num_vertices = None
    header_end = None
    prop_names = []

    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"):
            num_vertices = int(s.split()[-1])
        elif s.startswith("property"):
            prop_names.append(s.split()[-1])
        elif s == "end_header":
            header_end = idx + 1
            break

    if num_vertices is None or header_end is None:
        raise ValueError(f"Could not parse header for {path}")

    data_lines = lines[header_end:header_end + num_vertices]
    data = np.fromiter(
        (float(item) for line in data_lines for item in line.split()),
        dtype=np.float32,
        count=len(data_lines[0].split()) * num_vertices,
    ).reshape(num_vertices, -1)

    idxs = {name: i for i, name in enumerate(prop_names)}
    x = data[:, idxs["x"]]
    y = data[:, idxs["y"]]
    z = data[:, idxs["z"]]

    if {"red", "green", "blue"} <= idxs.keys():
        colors = data[:, [idxs["red"], idxs["green"],
                          idxs["blue"]]].astype(np.uint8)
    else:
        colors = np.full((num_vertices, 3), 180, dtype=np.uint8)
    return x, y, z, colors


def subsample(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, max_points: int):
    n = x.size
    if n <= max_points:
        return x, y, z, colors, 1
    idx = np.random.choice(n, max_points, replace=False)
    stride = int(np.ceil(n / max_points))
    return x[idx], y[idx], z[idx], colors[idx], stride


def infer_time_from_colors(colors: np.ndarray) -> np.ndarray:
    """Approximate a time-step per point based on nearest gain tint."""
    gains_sorted = sorted(GAIN_COLORS.keys())
    palette = np.array([GAIN_COLORS[g] for g in gains_sorted], dtype=np.float32)
    diffs = colors[:, None, :].astype(np.float32) - palette[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    nearest_idx = np.argmin(dist2, axis=1)
    return nearest_idx.astype(np.float32)  # 0,1,2,...


def st_dbscan(coords: np.ndarray, times: np.ndarray, eps_space: float, eps_time: float, min_samples: int) -> np.ndarray:
    """
    Simple ST-DBSCAN: spatial neighbors within eps_space AND temporal distance within eps_time.
    """
    n = coords.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)

    tree = BallTree(coords)
    neighbors = tree.query_radius(coords, r=eps_space)

    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neigh = [idx for idx in neighbors[i] if abs(times[idx] - times[i]) <= eps_time]
        if len(neigh) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = set(neigh)
        while seeds:
            pt = seeds.pop()
            if not visited[pt]:
                visited[pt] = True
                neigh_pt = [idx for idx in neighbors[pt] if abs(times[idx] - times[pt]) <= eps_time]
                if len(neigh_pt) >= min_samples:
                    seeds.update(neigh_pt)
            if labels[pt] == -1:
                labels[pt] = cluster_id
        cluster_id += 1

    return labels


def save_labels_csv(path: Path, coords: np.ndarray, labels: np.ndarray) -> None:
    arr = np.column_stack((coords, labels))
    header = "x,y,z,label"
    np.savetxt(path, arr, fmt="%.6f,%.6f,%.6f,%d", header=header, comments="")


def plot_labels_png(path: Path, coords: np.ndarray, labels: np.ndarray, colors: np.ndarray) -> None:
    if not HAS_MPL:
        print("matplotlib not installed; skipping plot.")
        return
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Map labels to colors; noise (-1) in gray
    unique = np.unique(labels)
    lut = {}
    for i, lbl in enumerate(unique):
        if lbl == -1:
            lut[lbl] = np.array([120, 120, 120], dtype=np.uint8)
        else:
            # reuse original color tint if available
            lut[lbl] = colors[labels == lbl][0] if (labels == lbl).any(
            ) else np.random.randint(0, 255, 3, dtype=np.uint8)
    label_colors = np.vstack([lut[l] for l in labels]
                             ).astype(np.float32) / 255.0
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=label_colors, s=0.5, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"ST-DBSCAN clusters ({path.name})")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def process_one(ply_path: Path, out_stem: str, eps: float, min_samples: int, max_points: int) -> None:
    x, y, z, colors = load_ply(ply_path)
    x, y, z, colors, stride = subsample(x, y, z, colors, max_points)
    coords = np.column_stack((x, y, z))
    times = infer_time_from_colors(colors)
    print(
        f"{ply_path.name}: using {coords.shape[0]:,} points (approx stride={stride})")

    labels = st_dbscan(coords, times, eps_space=eps, eps_time=EPS_TIME, min_samples=min_samples)
    unique, counts = np.unique(labels, return_counts=True)
    summary = dict(zip(unique.tolist(), counts.tolist()))
    print(f"{ply_path.name}: labels summary {summary}")

    csv_out = ply_path.with_name(f"{out_stem}_labels.csv")
    save_labels_csv(csv_out, coords, labels)
    print(f"labels CSV -> {csv_out.name}")

    png_out = ply_path.with_name(f"{out_stem}_labels.png")
    plot_labels_png(png_out, coords, labels, colors)
    print(f"plot -> {png_out.name}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir / "1.5_Folder"
    default_offset = default_dir / "frame_stack_v3.ply"
    default_flat = default_dir / "frame_stack_flat_v3.ply"

    p = argparse.ArgumentParser(
        description="Run ST-DBSCAN on stacked PLY point clouds (using gain colors as time steps).")
    p.add_argument("--offset", type=Path, default=default_offset,
                   help="Path to offset stack PLY.")
    p.add_argument("--flat", type=Path, default=default_flat,
                   help="Path to flat stack PLY.")
    args = p.parse_args()

    for ply_path, stem_suffix in [(args.offset, "frame_stack_v3_dbscan"), (args.flat, "frame_stack_flat_v3_dbscan")]:
        if not ply_path.exists():
            print(f"skip: {ply_path} not found")
            continue
        process_one(ply_path, stem_suffix, eps=EPS_SPACE,
                    min_samples=MIN_SAMPLES, max_points=MAX_POINTS)


if __name__ == "__main__":
    main()

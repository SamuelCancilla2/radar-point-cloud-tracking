"""Data writers for PLY and CSV formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import PointCloud


def write_ply(path: Path, cloud: PointCloud) -> None:
    """
    Write point cloud to ASCII PLY file with RGB colors.

    Parameters
    ----------
    path : Path
        Output file path.
    cloud : PointCloud
        Point cloud data with x, y, z and optional colors.
    """
    num_points = cloud.size
    colors = cloud.colors
    if colors is None:
        colors = np.full((num_points, 3), 180, dtype=np.uint8)

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

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        for xp, yp, zp, (r, g, b) in zip(cloud.x, cloud.y, cloud.z, colors, strict=False):
            fh.write(f"{xp:.6f} {yp:.6f} {zp:.6f} {r} {g} {b}\n")


def write_cartesian_csv(path: Path, cloud: PointCloud) -> None:
    """
    Write point cloud to CSV with x, y, z columns.

    Parameters
    ----------
    path : Path
        Output file path.
    cloud : PointCloud
        Point cloud data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"x": cloud.x, "y": cloud.y, "z": cloud.z})
    df.to_csv(path, index=False)


def write_labels_csv(path: Path, coords: np.ndarray, labels: np.ndarray) -> None:
    """
    Write labeled coordinates to CSV.

    Parameters
    ----------
    path : Path
        Output file path.
    coords : np.ndarray
        Nx3 array of coordinates.
    labels : np.ndarray
        Array of cluster labels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack((coords, labels))
    header = "x,y,z,label"
    np.savetxt(path, arr, fmt="%.6f,%.6f,%.6f,%d", header=header, comments="")

"""Consolidated data loaders for radar CSV and PLY point cloud files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from ..config import RadarConfig


@dataclass
class RadarSweep:
    """Container for radar sweep data."""

    angles_rad: np.ndarray
    ranges: np.ndarray
    intensities: np.ndarray
    scale: np.ndarray
    gain: Optional[int] = None
    source_path: Optional[Path] = None


@dataclass
class PointCloud:
    """Container for 3D point cloud data."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    colors: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        """Return number of points."""
        return self.x.size

    def to_coords(self) -> np.ndarray:
        """Return coordinates as Nx3 array."""
        return np.column_stack((self.x, self.y, self.z))


def load_radar_csv(
    path: Path,
    config: Optional[RadarConfig] = None,
) -> RadarSweep:
    """
    Load radar sweep data from CSV file.

    Parameters
    ----------
    path : Path
        Path to the radar CSV file.
    config : RadarConfig, optional
        Radar configuration. Uses defaults if not provided.

    Returns
    -------
    RadarSweep
        Container with angles, ranges, intensities, and scale.
    """
    if config is None:
        config = RadarConfig()

    col_names = ["Status", "Scale", "Range", "Gain", "Angle"] + [
        f"Echo_{i}" for i in range(config.num_echo_columns)
    ]

    df = pd.read_csv(path, header=None, names=col_names, skiprows=1, engine="c")
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    # Convert angle units to radians
    angles_rad = np.deg2rad(df["Angle"].to_numpy(np.float32) * config.angle_scale)

    # Extract echo/intensity data
    echo_data = df.iloc[:, 5:].fillna(0).to_numpy(np.float32)

    # Compute ranges based on scale
    max_ranges = df["Scale"].to_numpy(np.float32)
    num_bins = echo_data.shape[1]
    ranges = (max_ranges[:, None] / num_bins) * np.arange(num_bins, dtype=np.float32)

    # Extract gain if available
    gain = None
    if "Gain" in df.columns:
        gains = df["Gain"].unique()
        if len(gains) == 1:
            gain = int(gains[0])

    return RadarSweep(
        angles_rad=angles_rad,
        ranges=ranges,
        intensities=echo_data,
        scale=max_ranges,
        gain=gain,
        source_path=path,
    )


def load_radar_sweep_simple(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load radar sweep with simplified angle computation (uniform distribution).

    Parameters
    ----------
    path : Path
        Path to the radar CSV file.

    Returns
    -------
    tuple
        (angles_rad, intensities) arrays.
    """
    df = pd.read_csv(path, header=None, skiprows=1)
    intensities = df.iloc[:, 5:].to_numpy(dtype=np.float32)
    num_angles = len(df)
    angles_rad = np.linspace(0.0, 2 * np.pi, num_angles, endpoint=False, dtype=np.float32)
    return angles_rad, intensities


def load_cartesian_csv(path: Path) -> PointCloud:
    """
    Load Cartesian point cloud from CSV with x, y, z columns.

    Parameters
    ----------
    path : Path
        Path to the CSV file with x, y, z columns.

    Returns
    -------
    PointCloud
        Container with x, y, z coordinates.
    """
    df = pd.read_csv(path)
    col_map = {c.lower(): c for c in df.columns}

    x = df[col_map.get("x", df.columns[0])].to_numpy(np.float32)
    y = df[col_map.get("y", df.columns[1])].to_numpy(np.float32)
    z = df[col_map.get("z", df.columns[2])].to_numpy(np.float32)

    return PointCloud(x=x, y=y, z=z)


def load_ply(path: Path) -> PointCloud:
    """
    Load point cloud from ASCII PLY file.

    Parameters
    ----------
    path : Path
        Path to the PLY file.

    Returns
    -------
    PointCloud
        Container with x, y, z coordinates and optional RGB colors.
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

    data_lines = lines[header_end : header_end + num_vertices]
    if len(data_lines) < num_vertices:
        raise ValueError(f"Expected {num_vertices} vertices, found {len(data_lines)}")

    # Parse data into array
    num_cols = len(data_lines[0].split())
    data = np.fromiter(
        (float(item) for line in data_lines for item in line.split()),
        dtype=np.float32,
        count=num_cols * num_vertices,
    ).reshape(num_vertices, -1)

    # Map property names to indices
    prop_idx = {name: i for i, name in enumerate(prop_names)}

    try:
        x = data[:, prop_idx["x"]]
        y = data[:, prop_idx["y"]]
        z = data[:, prop_idx["z"]]
    except KeyError as exc:
        raise ValueError(f"PLY missing x/y/z properties: {path}") from exc

    # Extract colors if present
    colors = None
    if {"red", "green", "blue"} <= prop_idx.keys():
        r = data[:, prop_idx["red"]]
        g = data[:, prop_idx["green"]]
        b = data[:, prop_idx["blue"]]
        colors = np.stack([r, g, b], axis=1).astype(np.uint8)
    else:
        # Default gray color
        colors = np.full((num_vertices, 3), 180, dtype=np.uint8)

    return PointCloud(x=x, y=y, z=z, colors=colors)


def detect_csv_format(path: Path) -> Literal["radar", "cartesian"]:
    """
    Detect whether a CSV is radar sweep format or Cartesian point format.

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    str
        Either "radar" or "cartesian".
    """
    df_preview = pd.read_csv(path, nrows=2)
    lower_cols = [c.lower() for c in df_preview.columns]
    has_xyz_header = {"x", "y", "z"}.issubset(lower_cols)

    if has_xyz_header or (df_preview.shape[1] == 3 and df_preview.columns[0] != 0):
        return "cartesian"
    return "radar"


def load_points_auto(path: Path, config: Optional[RadarConfig] = None) -> PointCloud:
    """
    Auto-detect CSV format and load as point cloud.

    Parameters
    ----------
    path : Path
        Path to CSV file.
    config : RadarConfig, optional
        Radar configuration for radar CSV format.

    Returns
    -------
    PointCloud
        Point cloud data.
    """
    from .transforms import sweep_to_point_cloud

    fmt = detect_csv_format(path)
    if fmt == "cartesian":
        return load_cartesian_csv(path)

    sweep = load_radar_csv(path, config)
    return sweep_to_point_cloud(sweep)

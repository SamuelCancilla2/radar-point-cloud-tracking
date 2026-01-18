"""Point cloud building and stacking operations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import GainConfig, ProcessingConfig, RadarConfig
from ..core.loaders import PointCloud, detect_csv_format, load_cartesian_csv, load_radar_csv
from ..core.transforms import (
    apply_stride,
    gain_to_colors,
    sweep_to_points_simple,
)
from ..core.writers import write_ply


def find_gain_sweeps(directory: Path) -> Dict[int, Path]:
    """
    Discover gain-specific sweep CSVs in a directory.

    Parameters
    ----------
    directory : Path
        Directory to search.

    Returns
    -------
    dict
        Mapping of gain value to file path.
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


def load_points_from_csv(
    path: Path,
    config: Optional[ProcessingConfig] = None,
    radar_config: Optional[RadarConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load points from CSV, auto-detecting format.

    Parameters
    ----------
    path : Path
        CSV file path.
    config : ProcessingConfig, optional
        Processing configuration.
    radar_config : RadarConfig, optional
        Radar configuration.

    Returns
    -------
    tuple
        (x, y, z) point arrays.
    """
    if config is None:
        config = ProcessingConfig()
    if radar_config is None:
        radar_config = RadarConfig()

    fmt = detect_csv_format(path)

    if fmt == "cartesian":
        cloud = load_cartesian_csv(path)
        return cloud.x, cloud.y, cloud.z

    # Load as radar sweep
    import pandas as pd
    df = pd.read_csv(path, header=None, skiprows=1)
    intensities = df.iloc[:, 5:].to_numpy(dtype=np.float32)
    num_angles = len(df)
    angles_rad = np.linspace(0.0, 2 * np.pi, num_angles, endpoint=False, dtype=np.float32)

    return sweep_to_points_simple(
        angles_rad,
        intensities,
        range_bin_width=radar_config.range_bin_width_m,
        range_start=radar_config.range_start_m,
        min_intensity=config.intensity_threshold,
        stride=config.point_stride,
    )


def apply_gain_colors(
    z: np.ndarray,
    gain: int,
    gain_config: Optional[GainConfig] = None,
) -> np.ndarray:
    """
    Apply gain-specific colors to points.

    Parameters
    ----------
    z : np.ndarray
        Z values (used for sizing).
    gain : int
        Gain value.
    gain_config : GainConfig, optional
        Gain configuration.

    Returns
    -------
    np.ndarray
        Nx3 RGB array.
    """
    if gain_config is None:
        gain_config = GainConfig()
    return gain_to_colors(z, gain, gain_config.colors)


def combine_clouds(
    clouds: List[Tuple[int, PointCloud]],
    apply_offsets: bool = False,
    gain_config: Optional[GainConfig] = None,
) -> PointCloud:
    """
    Combine multiple point clouds with optional Z offsets.

    Parameters
    ----------
    clouds : list
        List of (gain, PointCloud) tuples.
    apply_offsets : bool
        If True, apply gain-specific Z offsets.
    gain_config : GainConfig, optional
        Gain configuration.

    Returns
    -------
    PointCloud
        Combined point cloud.
    """
    if gain_config is None:
        gain_config = GainConfig()

    all_x = []
    all_y = []
    all_z = []
    all_colors = []

    for gain, cloud in clouds:
        all_x.append(cloud.x)
        all_y.append(cloud.y)

        if apply_offsets:
            offset = gain_config.z_offsets.get(gain, 0.0)
            all_z.append(cloud.z + offset)
        else:
            all_z.append(cloud.z)

        if cloud.colors is not None:
            all_colors.append(cloud.colors)
        else:
            all_colors.append(apply_gain_colors(cloud.z, gain, gain_config))

    return PointCloud(
        x=np.concatenate(all_x),
        y=np.concatenate(all_y),
        z=np.concatenate(all_z),
        colors=np.concatenate(all_colors),
    )


def build_stacked_clouds(
    sweep_dir: Path,
    output_dir: Path,
    config: Optional[ProcessingConfig] = None,
    gain_config: Optional[GainConfig] = None,
    radar_config: Optional[RadarConfig] = None,
    generate_flat: bool = True,
    generate_offset: bool = True,
    name_prefix: str = "frame_stack",
) -> Dict[str, Path]:
    """
    Build stacked point clouds from gain-specific CSVs.

    Parameters
    ----------
    sweep_dir : Path
        Directory containing gain CSVs.
    output_dir : Path
        Output directory.
    config : ProcessingConfig, optional
        Processing configuration.
    gain_config : GainConfig, optional
        Gain configuration.
    radar_config : RadarConfig, optional
        Radar configuration.
    generate_flat : bool
        Generate flat (no offset) stack.
    generate_offset : bool
        Generate offset (separated layers) stack.
    name_prefix : str
        Output file name prefix.

    Returns
    -------
    dict
        Mapping of variant name to output path.
    """
    if config is None:
        config = ProcessingConfig()
    if gain_config is None:
        gain_config = GainConfig()
    if radar_config is None:
        radar_config = RadarConfig()

    sweep_files = find_gain_sweeps(sweep_dir)
    clouds: List[Tuple[int, PointCloud]] = []

    for gain, sweep_path in sweep_files.items():
        x, y, z = load_points_from_csv(sweep_path, config, radar_config)
        base_points = x.size

        # Auto-raise stride if file is very large
        gain_stride = max(config.point_stride, int(np.ceil(base_points / config.max_points_per_gain)))
        if gain_stride > 1:
            x = x[::gain_stride]
            y = y[::gain_stride]
            z = z[::gain_stride]

        colors = apply_gain_colors(z, gain, gain_config)
        clouds.append((gain, PointCloud(x=x, y=y, z=z, colors=colors)))

        print(f"gain {gain}: {x.size:,} points (stride={gain_stride})")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if generate_offset:
        offset_cloud = combine_clouds(clouds, apply_offsets=True, gain_config=gain_config)
        # Apply stack stride
        stack_stride = max(1, int(np.ceil(offset_cloud.size / config.max_points_stack)))
        if stack_stride > 1:
            offset_cloud = apply_stride(offset_cloud, stack_stride)

        offset_path = output_dir / f"{name_prefix}_v3.ply"
        write_ply(offset_path, offset_cloud)
        outputs["offset"] = offset_path
        print(f"Offset stack: {offset_cloud.size:,} points -> {offset_path.name}")

    if generate_flat:
        flat_cloud = combine_clouds(clouds, apply_offsets=False, gain_config=gain_config)
        # Apply stack stride
        stack_stride = max(1, int(np.ceil(flat_cloud.size / config.max_points_stack)))
        if stack_stride > 1:
            flat_cloud = apply_stride(flat_cloud, stack_stride)

        flat_path = output_dir / f"{name_prefix}_flat_v3.ply"
        write_ply(flat_path, flat_cloud)
        outputs["flat"] = flat_path
        print(f"Flat stack: {flat_cloud.size:,} points -> {flat_path.name}")

    return outputs

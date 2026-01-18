"""Coordinate transformations and point cloud operations."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..config import ProcessingConfig, RadarConfig
from .loaders import PointCloud, RadarSweep


def polar_to_cartesian(
    angles_rad: np.ndarray,
    ranges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert polar coordinates to Cartesian.

    Parameters
    ----------
    angles_rad : np.ndarray
        Angles in radians, shape (N,).
    ranges : np.ndarray
        Range values, shape (N, M) where M is number of range bins.

    Returns
    -------
    tuple
        (x, y) coordinate arrays, each shape (N, M).
    """
    x = ranges * np.cos(angles_rad[:, None])
    y = ranges * np.sin(angles_rad[:, None])
    return x, y


def sweep_to_point_cloud(
    sweep: RadarSweep,
    config: Optional[ProcessingConfig] = None,
    radar_config: Optional[RadarConfig] = None,
) -> PointCloud:
    """
    Convert radar sweep to point cloud with intensity filtering.

    Parameters
    ----------
    sweep : RadarSweep
        Radar sweep data.
    config : ProcessingConfig, optional
        Processing configuration.
    radar_config : RadarConfig, optional
        Radar configuration.

    Returns
    -------
    PointCloud
        Point cloud with x, y coordinates and z as intensity.
    """
    if config is None:
        config = ProcessingConfig()
    if radar_config is None:
        radar_config = RadarConfig()

    x, y = polar_to_cartesian(sweep.angles_rad, sweep.ranges)
    z = sweep.intensities

    # Apply intensity threshold
    mask = z > config.intensity_threshold
    x_pts = x[mask]
    y_pts = y[mask]
    z_pts = z[mask]

    # Apply stride
    if config.point_stride > 1:
        x_pts = x_pts[:: config.point_stride]
        y_pts = y_pts[:: config.point_stride]
        z_pts = z_pts[:: config.point_stride]

    return PointCloud(x=x_pts, y=y_pts, z=z_pts)


def sweep_to_points_simple(
    angles_rad: np.ndarray,
    intensities: np.ndarray,
    range_bin_width: float = 0.5,
    range_start: float = 0.0,
    min_intensity: float = 0.0,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert sweep to points using uniform range bins.

    Parameters
    ----------
    angles_rad : np.ndarray
        Angles in radians.
    intensities : np.ndarray
        Intensity matrix (angles x bins).
    range_bin_width : float
        Width of each range bin in meters.
    range_start : float
        Starting range in meters.
    min_intensity : float
        Minimum intensity threshold.
    stride : int
        Subsampling stride.

    Returns
    -------
    tuple
        (x, y, z) point arrays.
    """
    num_bins = intensities.shape[1]
    ranges = range_start + np.arange(num_bins, dtype=np.float32) * range_bin_width
    range_grid = ranges[None, :]

    cos_angles = np.cos(angles_rad)[:, None].astype(np.float32)
    sin_angles = np.sin(angles_rad)[:, None].astype(np.float32)

    x = range_grid * cos_angles
    y = range_grid * sin_angles
    z = intensities

    mask = z > min_intensity
    x_pts, y_pts, z_pts = x[mask], y[mask], z[mask]

    if stride > 1:
        x_pts = x_pts[::stride]
        y_pts = y_pts[::stride]
        z_pts = z_pts[::stride]

    return x_pts, y_pts, z_pts


def subsample_cloud(
    cloud: PointCloud,
    max_points: int,
) -> Tuple[PointCloud, int]:
    """
    Randomly subsample point cloud to maximum number of points.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    max_points : int
        Maximum number of points.

    Returns
    -------
    tuple
        (subsampled_cloud, stride) where stride is the approximate reduction factor.
    """
    n = cloud.size
    if n <= max_points:
        return cloud, 1

    idx = np.random.choice(n, max_points, replace=False)
    stride = int(np.ceil(n / max_points))

    new_colors = cloud.colors[idx] if cloud.colors is not None else None
    return PointCloud(
        x=cloud.x[idx],
        y=cloud.y[idx],
        z=cloud.z[idx],
        colors=new_colors,
    ), stride


def apply_stride(
    cloud: PointCloud,
    stride: int,
) -> PointCloud:
    """
    Apply regular stride subsampling to point cloud.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    stride : int
        Stride factor (keep every Nth point).

    Returns
    -------
    PointCloud
        Subsampled point cloud.
    """
    if stride <= 1:
        return cloud

    new_colors = cloud.colors[::stride] if cloud.colors is not None else None
    return PointCloud(
        x=cloud.x[::stride],
        y=cloud.y[::stride],
        z=cloud.z[::stride],
        colors=new_colors,
    )


def apply_z_offset(cloud: PointCloud, offset: float) -> PointCloud:
    """
    Apply vertical offset to point cloud z-values.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    offset : float
        Z offset to add.

    Returns
    -------
    PointCloud
        Point cloud with offset z-values.
    """
    return PointCloud(
        x=cloud.x,
        y=cloud.y,
        z=cloud.z + offset,
        colors=cloud.colors,
    )


def intensity_to_colors(values: np.ndarray) -> np.ndarray:
    """
    Map intensity values to grayscale RGB colors.

    Parameters
    ----------
    values : np.ndarray
        Intensity values (0-255 scale).

    Returns
    -------
    np.ndarray
        Nx3 RGB array.
    """
    clipped = np.clip(values, 0, 255).astype(np.uint8)
    return np.stack([clipped, clipped, clipped], axis=1)


def gain_to_colors(values: np.ndarray, gain: int, gain_colors: dict) -> np.ndarray:
    """
    Map points to constant gain color.

    Parameters
    ----------
    values : np.ndarray
        Array of values (used for sizing).
    gain : int
        Gain value.
    gain_colors : dict
        Mapping of gain to RGB tuple.

    Returns
    -------
    np.ndarray
        Nx3 RGB array.
    """
    rgb = np.array(gain_colors.get(gain, (180, 180, 180)), dtype=np.uint8)
    return np.repeat(rgb[None, :], values.size, axis=0)

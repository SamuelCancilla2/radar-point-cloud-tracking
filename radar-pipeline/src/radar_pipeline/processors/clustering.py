"""ST-DBSCAN clustering for spatio-temporal point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import BallTree

from ..config import ClusteringConfig, GainConfig
from ..core.loaders import PointCloud, load_ply
from ..core.transforms import subsample_cloud
from ..core.writers import write_labels_csv


def infer_time_from_colors(
    colors: np.ndarray,
    gain_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Approximate time-step per point based on nearest gain tint.

    Parameters
    ----------
    colors : np.ndarray
        Nx3 RGB color array.
    gain_colors : dict, optional
        Mapping of gain to RGB.

    Returns
    -------
    np.ndarray
        Time values (0, 1, 2, ...) based on color matching.
    """
    if gain_colors is None:
        gain_colors = GainConfig().colors

    gains_sorted = sorted(gain_colors.keys())
    palette = np.array([gain_colors[g] for g in gains_sorted], dtype=np.float32)

    diffs = colors[:, None, :].astype(np.float32) - palette[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    nearest_idx = np.argmin(dist2, axis=1)

    return nearest_idx.astype(np.float32)


def st_dbscan(
    coords: np.ndarray,
    times: np.ndarray,
    eps_space: float,
    eps_time: float,
    min_samples: int,
) -> np.ndarray:
    """
    Spatio-Temporal DBSCAN clustering.

    Points are neighbors if they are within eps_space spatially
    AND within eps_time temporally.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 coordinate array.
    times : np.ndarray
        Time values per point.
    eps_space : float
        Spatial neighborhood radius.
    eps_time : float
        Temporal neighborhood threshold.
    min_samples : int
        Minimum points to form a cluster.

    Returns
    -------
    np.ndarray
        Cluster labels (-1 for noise).
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

        # Filter neighbors by temporal distance
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


def cluster_point_cloud(
    cloud: PointCloud,
    config: Optional[ClusteringConfig] = None,
    gain_config: Optional[GainConfig] = None,
) -> np.ndarray:
    """
    Run ST-DBSCAN on point cloud using colors as time proxy.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud with colors.
    config : ClusteringConfig, optional
        Clustering configuration.
    gain_config : GainConfig, optional
        Gain configuration for time inference.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """
    if config is None:
        config = ClusteringConfig()
    if gain_config is None:
        gain_config = GainConfig()

    coords = cloud.to_coords()
    times = infer_time_from_colors(cloud.colors, gain_config.colors)

    return st_dbscan(
        coords,
        times,
        eps_space=config.eps_space,
        eps_time=config.eps_time,
        min_samples=config.min_samples,
    )


def process_ply_clustering(
    ply_path: Path,
    output_dir: Optional[Path] = None,
    config: Optional[ClusteringConfig] = None,
    gain_config: Optional[GainConfig] = None,
) -> Tuple[Path, np.ndarray]:
    """
    Load PLY, run clustering, and save results.

    Parameters
    ----------
    ply_path : Path
        Path to input PLY file.
    output_dir : Path, optional
        Output directory. Defaults to PLY parent directory.
    config : ClusteringConfig, optional
        Clustering configuration.
    gain_config : GainConfig, optional
        Gain configuration.

    Returns
    -------
    tuple
        (csv_path, labels) - path to output CSV and label array.
    """
    if config is None:
        config = ClusteringConfig()
    if gain_config is None:
        gain_config = GainConfig()
    if output_dir is None:
        output_dir = ply_path.parent

    cloud = load_ply(ply_path)

    # Subsample if needed
    cloud, stride = subsample_cloud(cloud, config.max_points)
    print(f"{ply_path.name}: using {cloud.size:,} points (approx stride={stride})")

    labels = cluster_point_cloud(cloud, config, gain_config)

    # Summary
    unique, counts = np.unique(labels, return_counts=True)
    summary = dict(zip(unique.tolist(), counts.tolist()))
    print(f"{ply_path.name}: labels summary {summary}")

    # Save results
    out_stem = f"{ply_path.stem}_dbscan"
    csv_path = output_dir / f"{out_stem}_labels.csv"
    write_labels_csv(csv_path, cloud.to_coords(), labels)
    print(f"Labels CSV -> {csv_path.name}")

    return csv_path, labels

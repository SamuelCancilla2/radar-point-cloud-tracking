"""Point cloud visualization and plotting functions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..core.loaders import PointCloud

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


def check_matplotlib() -> None:
    """Raise error if matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting but is not installed.")


def labels_to_colors(
    labels: np.ndarray,
    original_colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert cluster labels to colors.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels (-1 for noise).
    original_colors : np.ndarray, optional
        Original point colors to use for clusters.

    Returns
    -------
    np.ndarray
        Nx3 RGB array.
    """
    unique = np.unique(labels)
    lut = {}

    for lbl in unique:
        if lbl == -1:
            lut[lbl] = np.array([120, 120, 120], dtype=np.uint8)
        elif original_colors is not None and (labels == lbl).any():
            lut[lbl] = original_colors[labels == lbl][0]
        else:
            # Generate deterministic color from label
            np.random.seed(int(lbl))
            lut[lbl] = np.random.randint(0, 255, 3, dtype=np.uint8)

    return np.vstack([lut[l] for l in labels]).astype(np.uint8)


def plot_point_cloud(
    path: Path,
    cloud: PointCloud,
    title: str = "Point Cloud",
    max_points: int = 1_000_000,
    alpha: float = 0.5,
    marker_size: float = 1.0,
    dpi: int = 200,
) -> None:
    """
    Save 3D scatter plot of point cloud.

    Parameters
    ----------
    path : Path
        Output PNG path.
    cloud : PointCloud
        Point cloud to plot.
    title : str
        Plot title.
    max_points : int
        Maximum points to plot (subsampled if larger).
    alpha : float
        Marker opacity (0-1).
    marker_size : float
        Marker size.
    dpi : int
        Output resolution.
    """
    check_matplotlib()

    x, y, z = cloud.x, cloud.y, cloud.z
    colors = cloud.colors

    n_points = x.size
    plot_stride = max(1, int(np.ceil(n_points / max_points)))

    if plot_stride > 1:
        x = x[::plot_stride]
        y = y[::plot_stride]
        z = z[::plot_stride]
        if colors is not None:
            colors = colors[::plot_stride]
        print(f"Plot subsample: {x.size:,} points (stride={plot_stride})")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if colors is not None:
        scatter_colors = colors.astype(np.float32) / 255.0
    else:
        scatter_colors = None

    ax.scatter(x, y, z, c=scatter_colors, s=marker_size, alpha=alpha)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z / Intensity")
    ax.set_title(title)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_labeled_cloud(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    original_colors: Optional[np.ndarray] = None,
    title: str = "Clustered Point Cloud",
    max_points: int = 1_000_000,
    alpha: float = 0.5,
    marker_size: float = 0.5,
    dpi: int = 200,
) -> None:
    """
    Save 3D scatter plot of labeled point cloud.

    Parameters
    ----------
    path : Path
        Output PNG path.
    coords : np.ndarray
        Nx3 coordinate array.
    labels : np.ndarray
        Cluster labels.
    original_colors : np.ndarray, optional
        Original point colors for cluster coloring.
    title : str
        Plot title.
    max_points : int
        Maximum points to plot.
    alpha : float
        Marker opacity.
    marker_size : float
        Marker size.
    dpi : int
        Output resolution.
    """
    check_matplotlib()

    n_points = coords.shape[0]
    plot_stride = max(1, int(np.ceil(n_points / max_points)))

    if plot_stride > 1:
        coords = coords[::plot_stride]
        labels = labels[::plot_stride]
        if original_colors is not None:
            original_colors = original_colors[::plot_stride]
        print(f"Plot subsample: {coords.shape[0]:,} points (stride={plot_stride})")

    label_colors = labels_to_colors(labels, original_colors)
    scatter_colors = label_colors.astype(np.float32) / 255.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=scatter_colors,
        s=marker_size,
        alpha=alpha,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_ply_preview(
    ply_path: Path,
    output_path: Optional[Path] = None,
    max_points: int = 1_000_000,
    alpha: float = 0.5,
) -> Path:
    """
    Generate PNG preview from PLY file.

    Parameters
    ----------
    ply_path : Path
        Input PLY file.
    output_path : Path, optional
        Output PNG path. Defaults to PLY path with .png extension.
    max_points : int
        Maximum points to plot.
    alpha : float
        Marker opacity.

    Returns
    -------
    Path
        Output PNG path.
    """
    from ..core.loaders import load_ply
    from ..core.transforms import subsample_cloud

    if output_path is None:
        output_path = ply_path.with_suffix(".png")

    cloud = load_ply(ply_path)
    cloud, stride = subsample_cloud(cloud, max_points)
    print(f"Loaded {cloud.size:,} points (approx stride={stride})")

    plot_point_cloud(
        output_path,
        cloud,
        title=ply_path.name,
        max_points=max_points,
        alpha=alpha,
    )
    print(f"PNG saved to {output_path}")

    return output_path

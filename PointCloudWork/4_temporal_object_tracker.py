#!/usr/bin/env python3
"""
Temporal Object Tracking System for Radar Point Clouds

This system:
1. Fuses multiple gain values (40, 50, 70/75) into unified point clouds per time frame
2. Applies temporal ST-DBSCAN for spatiotemporal clustering
3. Filters out land/stationary background
4. Classifies objects as buoys (stationary) or boats (moving)
5. Tracks objects with persistent IDs across frames

Usage:
    python 4_temporal_object_tracker.py --data-dir <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed; clustering will fail")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# =============================================================================
# Configuration
# =============================================================================

# Supported gain values - fused together per frame
# Note: User requested 40, 50, 70 but data has 75. We support both.
SUPPORTED_GAINS = {40, 50, 70, 75}

# Gain-specific colors for visualization
GAIN_COLORS = {
    40: (0, 114, 255),    # Blue
    50: (0, 200, 83),     # Green
    70: (255, 165, 0),    # Orange (for 70 if used)
    75: (255, 87, 34),    # Orange-red
}

# Radar parameters
ANGLE_SCALE = 360.0 / 8196.0  # Convert angle units to degrees
NUM_ECHO_COLUMNS = 1024

# Processing parameters
INTENSITY_THRESHOLD = 10.0      # Minimum intensity to keep a point
POINT_STRIDE = 4                # Keep every Nth point for efficiency
MAX_TIME_DIFF_MS = 2000         # Max time difference to consider same frame (ms)

# ST-DBSCAN parameters
EPS_SPACE = 8.0                 # Spatial neighborhood radius (meters)
EPS_TIME = 2.0                  # Temporal neighborhood (number of frames)
MIN_SAMPLES = 15                # Minimum points to form a cluster

# Land filtering parameters
LAND_PERSISTENCE_THRESHOLD = 0.8  # % of frames a point must appear to be "land"
LAND_GRID_RESOLUTION = 5.0        # Grid cell size for persistence analysis (meters)
LAND_MIN_INTENSITY = 100          # Minimum average intensity to be considered land

# Object classification parameters
STATIONARY_VELOCITY_THRESHOLD = 1.0  # m/frame - below this is "stationary" (buoy)
MOTION_HISTORY_FRAMES = 5            # Frames to average for velocity calculation

# Tracking parameters
MAX_ASSOCIATION_DISTANCE = 50.0   # Max distance to associate objects across frames
MAX_MISSED_FRAMES = 10            # Max frames an object can be missed before deletion


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RadarFrame:
    """A single radar frame with fused gain data."""
    timestamp: datetime
    timestamp_ms: int
    frame_id: int
    points: np.ndarray      # Shape: (N, 3) - x, y, intensity
    gains: np.ndarray       # Shape: (N,) - which gain each point came from

    @property
    def num_points(self) -> int:
        return self.points.shape[0]


@dataclass
class TrackedObject:
    """An object being tracked across frames."""
    object_id: int
    object_type: str            # "buoy", "boat", or "unknown"
    positions: List[np.ndarray] = field(default_factory=list)  # History of centroids
    frames_seen: List[int] = field(default_factory=list)       # Frame IDs where seen
    last_seen_frame: int = 0
    velocities: List[np.ndarray] = field(default_factory=list)
    color: Tuple[int, int, int] = (180, 180, 180)

    @property
    def centroid(self) -> np.ndarray:
        """Current position (last known)."""
        return self.positions[-1] if self.positions else np.array([0, 0])

    @property
    def average_velocity(self) -> float:
        """Average velocity magnitude over recent history."""
        if len(self.velocities) < 2:
            return 0.0
        recent = self.velocities[-MOTION_HISTORY_FRAMES:]
        return np.mean([np.linalg.norm(v) for v in recent])

    def predict_position(self, frames_ahead: int = 1) -> np.ndarray:
        """Predict position based on velocity."""
        if len(self.velocities) < 1:
            return self.centroid
        avg_vel = np.mean(self.velocities[-MOTION_HISTORY_FRAMES:], axis=0)
        return self.centroid + avg_vel * frames_ahead


@dataclass
class Cluster:
    """A cluster of points detected in a single frame."""
    cluster_id: int
    frame_id: int
    points: np.ndarray        # Shape: (N, 2) - x, y
    intensities: np.ndarray   # Shape: (N,)
    centroid: np.ndarray      # Shape: (2,) - x, y

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def mean_intensity(self) -> float:
        return float(np.mean(self.intensities))


# =============================================================================
# File Loading and Frame Building
# =============================================================================

def parse_timestamp(filename: str) -> Tuple[datetime, int]:
    """
    Parse timestamp from filename like '20250813_142602_181.csv'.
    Returns (datetime, total_milliseconds).
    """
    match = re.match(r"(\d{8})_(\d{6})_(\d{3})\.csv", filename)
    if not match:
        raise ValueError(f"Cannot parse timestamp from {filename}")

    date_str, time_str, ms_str = match.groups()
    dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    ms = int(ms_str)

    # Total milliseconds from epoch for easy comparison
    total_ms = int(dt.timestamp() * 1000) + ms

    return dt, total_ms


def load_radar_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load a radar CSV and convert to Cartesian coordinates.
    Returns (x, y, intensity, gain).
    """
    col_names = ["Status", "Scale", "Range", "Gain", "Angle"] + [f"Echo_{i}" for i in range(NUM_ECHO_COLUMNS)]

    try:
        df = pd.read_csv(path, header=None, names=col_names, skiprows=1, engine="c")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.array([]), np.array([]), np.array([]), 0

    if df.empty:
        return np.array([]), np.array([]), np.array([]), 0

    gain = int(df["Gain"].iloc[0])

    # Convert angles from radar units to radians
    angles_rad = np.deg2rad(df["Angle"].to_numpy(np.float32) * ANGLE_SCALE)

    # Get echo data
    echo_data = df.iloc[:, 5:].fillna(0).to_numpy(np.float32)

    # Get scale (max range) per angle
    max_ranges = df["Scale"].to_numpy(np.float32)
    num_bins = echo_data.shape[1]

    # Calculate range for each bin
    range_res = max_ranges[:, None] / num_bins
    ranges = range_res * np.arange(num_bins, dtype=np.float32)

    # Convert to Cartesian
    x = ranges * np.cos(angles_rad[:, None])
    y = ranges * np.sin(angles_rad[:, None])

    # Apply intensity threshold and flatten
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
    """
    Discover all CSV files organized by gain.
    Returns {gain: [list of paths sorted by timestamp]}.
    """
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

    # Sort by timestamp
    result = {}
    for gain, files in files_by_gain.items():
        files.sort(key=lambda x: x[0])
        result[gain] = [f[1] for f in files]

    return result


def group_files_by_frame(files_by_gain: Dict[int, List[Path]]) -> List[Dict[int, Path]]:
    """
    Group files across gains into frames based on timestamp proximity.
    Returns list of {gain: path} dicts, one per frame.
    """
    # Collect all files with timestamps
    all_files: List[Tuple[int, int, Path]] = []  # (timestamp_ms, gain, path)

    for gain, paths in files_by_gain.items():
        for path in paths:
            _, ts_ms = parse_timestamp(path.name)
            all_files.append((ts_ms, gain, path))

    all_files.sort(key=lambda x: x[0])

    # Group into frames
    frames: List[Dict[int, Path]] = []
    current_frame: Dict[int, Path] = {}
    frame_start_ts = None

    for ts_ms, gain, path in all_files:
        if frame_start_ts is None:
            frame_start_ts = ts_ms
            current_frame = {gain: path}
        elif ts_ms - frame_start_ts <= MAX_TIME_DIFF_MS:
            # Same frame - add if we don't have this gain yet
            if gain not in current_frame:
                current_frame[gain] = path
        else:
            # New frame
            if current_frame:
                frames.append(current_frame)
            frame_start_ts = ts_ms
            current_frame = {gain: path}

    # Don't forget the last frame
    if current_frame:
        frames.append(current_frame)

    return frames


def build_frame(frame_files: Dict[int, Path], frame_id: int) -> Optional[RadarFrame]:
    """
    Build a single RadarFrame by fusing data from multiple gains.
    Intensity is taken as absolute maximum across gains.
    """
    all_x, all_y, all_intensity, all_gains = [], [], [], []

    first_ts = None
    first_ts_ms = None

    for gain, path in sorted(frame_files.items()):
        if first_ts is None:
            first_ts, first_ts_ms = parse_timestamp(path.name)

        x, y, intensity, _ = load_radar_csv(path)
        if len(x) == 0:
            continue

        all_x.append(x)
        all_y.append(y)
        all_intensity.append(intensity)
        all_gains.append(np.full(len(x), gain, dtype=np.int32))

    if not all_x:
        return None

    # Concatenate all data
    x_combined = np.concatenate(all_x)
    y_combined = np.concatenate(all_y)
    intensity_combined = np.concatenate(all_intensity)
    gains_combined = np.concatenate(all_gains)

    points = np.column_stack([x_combined, y_combined, intensity_combined])

    return RadarFrame(
        timestamp=first_ts,
        timestamp_ms=first_ts_ms,
        frame_id=frame_id,
        points=points,
        gains=gains_combined
    )


# =============================================================================
# Land Filtering
# =============================================================================

def build_occupancy_grid(frames: List[RadarFrame], resolution: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an occupancy grid showing how often each cell is occupied.
    Returns (count_grid, intensity_sum_grid, (x_edges, y_edges)).
    """
    # Find bounds
    all_x = np.concatenate([f.points[:, 0] for f in frames])
    all_y = np.concatenate([f.points[:, 1] for f in frames])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    # Create grid
    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)

    count_grid = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.int32)
    intensity_grid = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64)

    for frame in frames:
        x = frame.points[:, 0]
        y = frame.points[:, 1]
        intensity = frame.points[:, 2]

        # Digitize to find grid cells
        x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, len(x_edges) - 2)
        y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, len(y_edges) - 2)

        # Update counts
        np.add.at(count_grid, (x_idx, y_idx), 1)
        np.add.at(intensity_grid, (x_idx, y_idx), intensity)

    return count_grid, intensity_grid, (x_edges, y_edges)


def identify_land_cells(count_grid: np.ndarray, intensity_grid: np.ndarray,
                        num_frames: int) -> np.ndarray:
    """
    Identify grid cells that are likely land based on persistence.
    Returns boolean mask of land cells.
    """
    # Calculate persistence (fraction of frames occupied)
    persistence = count_grid / max(num_frames, 1)

    # Calculate average intensity where occupied
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_intensity = np.where(count_grid > 0, intensity_grid / count_grid, 0)

    # Land is high persistence AND high intensity
    land_mask = (persistence >= LAND_PERSISTENCE_THRESHOLD) & (avg_intensity >= LAND_MIN_INTENSITY)

    return land_mask


def filter_land_from_frame(frame: RadarFrame, land_mask: np.ndarray,
                           edges: Tuple[np.ndarray, np.ndarray]) -> RadarFrame:
    """
    Remove land points from a frame.
    """
    x_edges, y_edges = edges

    x = frame.points[:, 0]
    y = frame.points[:, 1]

    # Find grid cells
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, land_mask.shape[0] - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, land_mask.shape[1] - 1)

    # Keep only non-land points
    keep_mask = ~land_mask[x_idx, y_idx]

    return RadarFrame(
        timestamp=frame.timestamp,
        timestamp_ms=frame.timestamp_ms,
        frame_id=frame.frame_id,
        points=frame.points[keep_mask],
        gains=frame.gains[keep_mask]
    )


# =============================================================================
# ST-DBSCAN Clustering
# =============================================================================

def st_dbscan(frames: List[RadarFrame], eps_space: float, eps_time: float,
              min_samples: int) -> Dict[int, List[Cluster]]:
    """
    Apply ST-DBSCAN clustering across all frames.
    Returns {frame_id: [list of Clusters]}.
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for clustering")

    # Combine all points with frame info
    all_points = []
    all_frame_ids = []
    point_offsets = [0]  # Track where each frame's points start

    for frame in frames:
        xy = frame.points[:, :2]  # x, y only for spatial clustering
        all_points.append(xy)
        all_frame_ids.extend([frame.frame_id] * len(xy))
        point_offsets.append(point_offsets[-1] + len(xy))

    if not all_points:
        return {}

    coords = np.vstack(all_points)
    frame_ids = np.array(all_frame_ids, dtype=np.float32)

    n = coords.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)

    # Build spatial index
    tree = BallTree(coords)
    spatial_neighbors = tree.query_radius(coords, r=eps_space)

    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # Find neighbors that are close in space AND time
        neigh = [idx for idx in spatial_neighbors[i]
                 if abs(frame_ids[idx] - frame_ids[i]) <= eps_time]

        if len(neigh) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = set(neigh)

        while seeds:
            pt = seeds.pop()
            if not visited[pt]:
                visited[pt] = True
                neigh_pt = [idx for idx in spatial_neighbors[pt]
                            if abs(frame_ids[idx] - frame_ids[pt]) <= eps_time]
                if len(neigh_pt) >= min_samples:
                    seeds.update(neigh_pt)
            if labels[pt] == -1:
                labels[pt] = cluster_id

        cluster_id += 1

    # Convert labels back to per-frame clusters
    clusters_by_frame: Dict[int, List[Cluster]] = defaultdict(list)

    for frame_idx, frame in enumerate(frames):
        start = point_offsets[frame_idx]
        end = point_offsets[frame_idx + 1]

        frame_labels = labels[start:end]
        frame_coords = coords[start:end]
        frame_intensities = frame.points[:, 2]

        unique_labels = set(frame_labels)
        unique_labels.discard(-1)  # Remove noise label

        for lbl in unique_labels:
            mask = frame_labels == lbl
            pts = frame_coords[mask]
            ints = frame_intensities[mask]

            cluster = Cluster(
                cluster_id=int(lbl),
                frame_id=frame.frame_id,
                points=pts,
                intensities=ints,
                centroid=np.mean(pts, axis=0)
            )
            clusters_by_frame[frame.frame_id].append(cluster)

    return dict(clusters_by_frame)


# =============================================================================
# Object Tracking
# =============================================================================

class ObjectTracker:
    """
    Track objects across frames using Hungarian algorithm for association.
    """

    def __init__(self):
        self.objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 1
        self.current_frame = 0

    def update(self, clusters: List[Cluster], frame_id: int) -> List[TrackedObject]:
        """
        Update tracker with new clusters from a frame.
        Returns list of currently tracked objects.
        """
        self.current_frame = frame_id

        if not clusters:
            # No detections - mark all objects as potentially lost
            return self._cleanup_lost_objects()

        if not self.objects:
            # First frame - initialize all clusters as new objects
            for cluster in clusters:
                self._create_object(cluster)
            return list(self.objects.values())

        # Associate clusters to existing objects using Hungarian algorithm
        active_objects = [obj for obj in self.objects.values()
                         if frame_id - obj.last_seen_frame <= MAX_MISSED_FRAMES]

        if not active_objects:
            # All objects lost - create new ones
            for cluster in clusters:
                self._create_object(cluster)
            return list(self.objects.values())

        # Build cost matrix
        cost_matrix = np.zeros((len(clusters), len(active_objects)))

        for i, cluster in enumerate(clusters):
            for j, obj in enumerate(active_objects):
                # Cost is distance from predicted position
                predicted_pos = obj.predict_position(frame_id - obj.last_seen_frame)
                cost_matrix[i, j] = np.linalg.norm(cluster.centroid - predicted_pos)

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Process assignments
        assigned_clusters = set()
        assigned_objects = set()

        for cluster_idx, obj_idx in zip(row_ind, col_ind):
            if cost_matrix[cluster_idx, obj_idx] <= MAX_ASSOCIATION_DISTANCE:
                cluster = clusters[cluster_idx]
                obj = active_objects[obj_idx]
                self._update_object(obj, cluster, frame_id)
                assigned_clusters.add(cluster_idx)
                assigned_objects.add(obj.object_id)

        # Create new objects for unassigned clusters
        for i, cluster in enumerate(clusters):
            if i not in assigned_clusters:
                self._create_object(cluster)

        return self._cleanup_lost_objects()

    def _create_object(self, cluster: Cluster) -> TrackedObject:
        """Create a new tracked object."""
        obj = TrackedObject(
            object_id=self.next_object_id,
            object_type="unknown",
            positions=[cluster.centroid.copy()],
            frames_seen=[cluster.frame_id],
            last_seen_frame=cluster.frame_id,
            velocities=[np.array([0.0, 0.0])],
            color=self._generate_color(self.next_object_id)
        )
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1
        return obj

    def _update_object(self, obj: TrackedObject, cluster: Cluster, frame_id: int) -> None:
        """Update an existing object with new observation."""
        # Calculate velocity
        if obj.positions:
            frames_elapsed = frame_id - obj.last_seen_frame
            if frames_elapsed > 0:
                velocity = (cluster.centroid - obj.positions[-1]) / frames_elapsed
                obj.velocities.append(velocity)

        obj.positions.append(cluster.centroid.copy())
        obj.frames_seen.append(frame_id)
        obj.last_seen_frame = frame_id

        # Update classification based on motion
        obj.object_type = self._classify_object(obj)

    def _classify_object(self, obj: TrackedObject) -> str:
        """Classify object as buoy, boat, or unknown based on motion."""
        if len(obj.velocities) < MOTION_HISTORY_FRAMES:
            return "unknown"

        avg_velocity = obj.average_velocity

        if avg_velocity < STATIONARY_VELOCITY_THRESHOLD:
            return "buoy"
        else:
            return "boat"

    def _cleanup_lost_objects(self) -> List[TrackedObject]:
        """Remove objects that have been lost for too long."""
        to_remove = []
        for obj_id, obj in self.objects.items():
            if self.current_frame - obj.last_seen_frame > MAX_MISSED_FRAMES:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.objects[obj_id]

        return list(self.objects.values())

    def _generate_color(self, obj_id: int) -> Tuple[int, int, int]:
        """Generate a unique color for an object."""
        # Use golden ratio for good color distribution
        hue = (obj_id * 0.618033988749895) % 1.0
        # Convert HSV to RGB (simplified)
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        q = 1 - f

        if h_i == 0:
            r, g, b = 1, f, 0
        elif h_i == 1:
            r, g, b = q, 1, 0
        elif h_i == 2:
            r, g, b = 0, 1, f
        elif h_i == 3:
            r, g, b = 0, q, 1
        elif h_i == 4:
            r, g, b = f, 0, 1
        else:
            r, g, b = 1, 0, q

        return (int(r * 255), int(g * 255), int(b * 255))


# =============================================================================
# Visualization
# =============================================================================

def plot_frame_with_objects(frame: RadarFrame, clusters: List[Cluster],
                            objects: List[TrackedObject], output_path: Path,
                            land_mask: Optional[np.ndarray] = None,
                            edges: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
    """Plot a single frame with detected objects."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Raw point cloud colored by gain
    ax1 = axes[0]
    gain_cmap = {40: 'blue', 50: 'green', 70: 'orange', 75: 'red'}

    for gain in np.unique(frame.gains):
        mask = frame.gains == gain
        pts = frame.points[mask]
        color = gain_cmap.get(gain, 'gray')
        ax1.scatter(pts[:, 0], pts[:, 1], c=color, s=0.5, alpha=0.5, label=f'Gain {gain}')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Frame {frame.frame_id}: Raw Points by Gain')
    ax1.legend(markerscale=10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Right: Tracked objects
    ax2 = axes[1]

    # Plot background points as gray
    ax2.scatter(frame.points[:, 0], frame.points[:, 1], c='lightgray', s=0.5, alpha=0.3)

    # Plot clusters/objects with colors
    buoy_patches = []
    boat_patches = []

    for obj in objects:
        if obj.last_seen_frame != frame.frame_id:
            continue

        # Find matching cluster
        for cluster in clusters:
            if np.linalg.norm(cluster.centroid - obj.centroid) < 5:
                color = np.array(obj.color) / 255.0
                ax2.scatter(cluster.points[:, 0], cluster.points[:, 1],
                           c=[color], s=2, alpha=0.8)

                # Draw object ID and type
                label = f"{obj.object_type[0].upper()}{obj.object_id}"
                ax2.annotate(label, obj.centroid, fontsize=8,
                            ha='center', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # Draw velocity vector for boats
                if obj.object_type == "boat" and len(obj.velocities) > 0:
                    vel = np.mean(obj.velocities[-3:], axis=0) * 5  # Scale for visibility
                    ax2.arrow(obj.centroid[0], obj.centroid[1], vel[0], vel[1],
                             head_width=3, head_length=2, fc='red', ec='red')
                break

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Frame {frame.frame_id}: Tracked Objects')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Legend
    buoy_patch = mpatches.Patch(color='green', label='Buoy (stationary)')
    boat_patch = mpatches.Patch(color='red', label='Boat (moving)')
    ax2.legend(handles=[buoy_patch, boat_patch])

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tracking_summary(all_objects: List[TrackedObject], output_path: Path) -> None:
    """Plot summary of all tracked objects and their trajectories."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Trajectories
    ax1 = axes[0]

    buoys = [o for o in all_objects if o.object_type == "buoy"]
    boats = [o for o in all_objects if o.object_type == "boat"]
    unknown = [o for o in all_objects if o.object_type == "unknown"]

    for obj in buoys:
        if len(obj.positions) > 1:
            positions = np.array(obj.positions)
            ax1.plot(positions[:, 0], positions[:, 1], 'go-', markersize=4, alpha=0.7)
            ax1.annotate(f'B{obj.object_id}', positions[-1], fontsize=8)

    for obj in boats:
        if len(obj.positions) > 1:
            positions = np.array(obj.positions)
            ax1.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=2, alpha=0.7)
            ax1.scatter(positions[:, 0], positions[:, 1], c='red', s=10)
            ax1.annotate(f'V{obj.object_id}', positions[-1], fontsize=8)

    for obj in unknown:
        if len(obj.positions) > 1:
            positions = np.array(obj.positions)
            ax1.plot(positions[:, 0], positions[:, 1], 'b--', alpha=0.5)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Object Trajectories')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Buoys', 'Boats', 'Unknown'])

    # Right: Statistics
    ax2 = axes[1]

    categories = ['Buoys', 'Boats', 'Unknown']
    counts = [len(buoys), len(boats), len(unknown)]
    colors = ['green', 'red', 'blue']

    bars = ax2.bar(categories, counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Object Classification Summary')

    # Add count labels
    for bar, count in zip(bars, counts):
        ax2.annotate(str(count), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_tracking_results(tracker: ObjectTracker, clusters_by_frame: Dict[int, List[Cluster]],
                          output_dir: Path) -> None:
    """Save tracking results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save object summary
    objects_data = []
    for obj in tracker.objects.values():
        objects_data.append({
            'object_id': obj.object_id,
            'object_type': obj.object_type,
            'num_frames_seen': len(obj.frames_seen),
            'first_frame': min(obj.frames_seen) if obj.frames_seen else -1,
            'last_frame': max(obj.frames_seen) if obj.frames_seen else -1,
            'avg_velocity': obj.average_velocity,
            'final_x': obj.centroid[0],
            'final_y': obj.centroid[1],
        })

    objects_df = pd.DataFrame(objects_data)
    objects_df.to_csv(output_dir / "tracked_objects.csv", index=False)
    print(f"Saved object summary to {output_dir / 'tracked_objects.csv'}")

    # Save trajectories
    trajectories = []
    for obj in tracker.objects.values():
        for i, (pos, frame_id) in enumerate(zip(obj.positions, obj.frames_seen)):
            trajectories.append({
                'object_id': obj.object_id,
                'object_type': obj.object_type,
                'frame_id': frame_id,
                'x': pos[0],
                'y': pos[1],
            })

    traj_df = pd.DataFrame(trajectories)
    traj_df.to_csv(output_dir / "trajectories.csv", index=False)
    print(f"Saved trajectories to {output_dir / 'trajectories.csv'}")

    # Save cluster details per frame
    clusters_data = []
    for frame_id, clusters in clusters_by_frame.items():
        for cluster in clusters:
            clusters_data.append({
                'frame_id': frame_id,
                'cluster_id': cluster.cluster_id,
                'num_points': cluster.num_points,
                'centroid_x': cluster.centroid[0],
                'centroid_y': cluster.centroid[1],
                'mean_intensity': cluster.mean_intensity,
            })

    clusters_df = pd.DataFrame(clusters_data)
    clusters_df.to_csv(output_dir / "clusters.csv", index=False)
    print(f"Saved clusters to {output_dir / 'clusters.csv'}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(data_dir: Path, output_dir: Path, max_frames: int = 0,
                 skip_land_filter: bool = False, visualize: bool = True,
                 eps_space: float = EPS_SPACE, eps_time: float = EPS_TIME,
                 min_samples: int = MIN_SAMPLES, intensity_threshold: float = INTENSITY_THRESHOLD) -> None:
    """
    Run the complete tracking pipeline.

    Args:
        data_dir: Directory containing gain_40/, gain_50/, gain_75/ subdirs
        output_dir: Directory to save results
        max_frames: Maximum frames to process (0 = all)
        skip_land_filter: Skip land filtering step
        visualize: Generate visualization images
        eps_space: Spatial clustering radius
        eps_time: Temporal clustering window
        min_samples: Minimum points per cluster
        intensity_threshold: Minimum intensity to keep a point
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TEMPORAL OBJECT TRACKING PIPELINE")
    print("=" * 60)

    # Step 1: Discover and group files
    print("\n[1/6] Discovering data files...")
    files_by_gain = discover_files(data_dir)

    if not files_by_gain:
        print("ERROR: No valid data files found!")
        return

    for gain, files in files_by_gain.items():
        print(f"  Gain {gain}: {len(files)} files")

    # Step 2: Group into frames
    print("\n[2/6] Grouping files into temporal frames...")
    frame_files = group_files_by_frame(files_by_gain)
    print(f"  Found {len(frame_files)} frames")

    if max_frames > 0:
        frame_files = frame_files[:max_frames]
        print(f"  Processing first {len(frame_files)} frames")

    # Step 3: Build frames
    print("\n[3/6] Building point cloud frames...")
    frames: List[RadarFrame] = []

    for i, ff in enumerate(frame_files):
        frame = build_frame(ff, i)
        if frame:
            frames.append(frame)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(frame_files)} frames...")

    print(f"  Built {len(frames)} frames")
    total_points = sum(f.num_points for f in frames)
    print(f"  Total points: {total_points:,}")

    # Step 4: Land filtering
    if not skip_land_filter and len(frames) > 10:
        print("\n[4/6] Building land filter...")
        count_grid, intensity_grid, edges = build_occupancy_grid(frames, LAND_GRID_RESOLUTION)
        land_mask = identify_land_cells(count_grid, intensity_grid, len(frames))
        land_cells = np.sum(land_mask)
        print(f"  Identified {land_cells} land cells")

        print("  Filtering land from frames...")
        filtered_frames = []
        for frame in frames:
            filtered = filter_land_from_frame(frame, land_mask, edges)
            filtered_frames.append(filtered)

        points_removed = total_points - sum(f.num_points for f in filtered_frames)
        print(f"  Removed {points_removed:,} land points ({100*points_removed/total_points:.1f}%)")
        frames = filtered_frames
    else:
        print("\n[4/6] Skipping land filter")
        land_mask = None
        edges = None

    # Step 5: Clustering
    print("\n[5/6] Running ST-DBSCAN clustering...")
    clusters_by_frame = st_dbscan(frames, eps_space, eps_time, min_samples)

    total_clusters = sum(len(c) for c in clusters_by_frame.values())
    print(f"  Found {total_clusters} clusters across {len(clusters_by_frame)} frames")

    # Step 6: Object tracking
    print("\n[6/6] Tracking objects...")
    tracker = ObjectTracker()

    for frame in frames:
        clusters = clusters_by_frame.get(frame.frame_id, [])
        objects = tracker.update(clusters, frame.frame_id)

        if frame.frame_id % 50 == 0:
            print(f"  Frame {frame.frame_id}: {len(objects)} active objects")

    # Final statistics
    print("\n" + "=" * 60)
    print("TRACKING RESULTS")
    print("=" * 60)

    buoys = [o for o in tracker.objects.values() if o.object_type == "buoy"]
    boats = [o for o in tracker.objects.values() if o.object_type == "boat"]
    unknown = [o for o in tracker.objects.values() if o.object_type == "unknown"]

    print(f"  Total objects tracked: {len(tracker.objects)}")
    print(f"  Buoys (stationary): {len(buoys)}")
    print(f"  Boats (moving): {len(boats)}")
    print(f"  Unknown: {len(unknown)}")

    # Save results
    print("\nSaving results...")
    save_tracking_results(tracker, clusters_by_frame, output_dir)

    # Visualization
    if visualize and HAS_MPL:
        print("\nGenerating visualizations...")
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Plot a few sample frames
        sample_frames = frames[::max(1, len(frames)//10)]  # Every 10th frame
        for frame in sample_frames:
            clusters = clusters_by_frame.get(frame.frame_id, [])
            objects = [o for o in tracker.objects.values()
                      if frame.frame_id in o.frames_seen]

            plot_frame_with_objects(
                frame, clusters, objects,
                viz_dir / f"frame_{frame.frame_id:04d}.png",
                land_mask, edges
            )

        print(f"  Saved {len(sample_frames)} frame visualizations")

        # Plot tracking summary
        plot_tracking_summary(list(tracker.objects.values()),
                            output_dir / "tracking_summary.png")
        print("  Saved tracking summary")

    print("\nPipeline complete!")
    print(f"Results saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track objects in radar point cloud time series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 4_temporal_object_tracker.py --data-dir ./data --output-dir ./results
  python 4_temporal_object_tracker.py --data-dir ./data --output-dir ./results --max-frames 100
  python 4_temporal_object_tracker.py --data-dir ./data --output-dir ./results --no-land-filter
        """
    )

    script_dir = Path(__file__).resolve().parent
    default_data = script_dir.parent / "(.125NM)data_pattern3(.125NM)"
    default_output = script_dir / "tracking_results"

    parser.add_argument(
        "--data-dir", type=Path, default=default_data,
        help=f"Directory containing gain subdirectories (default: {default_data})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_output,
        help=f"Output directory for results (default: {default_output})"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Maximum frames to process (0 = all)"
    )
    parser.add_argument(
        "--no-land-filter", action="store_true",
        help="Skip land filtering step"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization generation"
    )

    # Allow overriding key parameters
    parser.add_argument("--eps-space", type=float, default=EPS_SPACE,
                       help=f"Spatial clustering radius (default: {EPS_SPACE})")
    parser.add_argument("--eps-time", type=float, default=EPS_TIME,
                       help=f"Temporal clustering window (default: {EPS_TIME})")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                       help=f"Min points per cluster (default: {MIN_SAMPLES})")
    parser.add_argument("--intensity-threshold", type=float, default=INTENSITY_THRESHOLD,
                       help=f"Minimum intensity threshold (default: {INTENSITY_THRESHOLD})")

    args = parser.parse_args()

    # Pass CLI args to pipeline via keyword arguments
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        skip_land_filter=args.no_land_filter,
        visualize=not args.no_viz,
        eps_space=args.eps_space,
        eps_time=args.eps_time,
        min_samples=args.min_samples,
        intensity_threshold=args.intensity_threshold
    )


if __name__ == "__main__":
    main()

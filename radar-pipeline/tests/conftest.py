"""Pytest fixtures for radar pipeline tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_radar_csv(tmp_path: Path) -> Path:
    """Create a minimal valid radar CSV file."""
    # Header + 2 data rows
    header = "Status,Scale,Range,Gain,Angle," + ",".join(f"Echo_{i}" for i in range(1024))
    row1 = "1,496,3,75,10," + ",".join(["128"] * 1024)
    row2 = "1,496,3,75,20," + ",".join(["64"] * 1024)

    csv_content = f"{header}\n{row1}\n{row2}"
    path = tmp_path / "test_radar.csv"
    path.write_text(csv_content)
    return path


@pytest.fixture
def sample_cartesian_csv(tmp_path: Path) -> Path:
    """Create a minimal Cartesian CSV file."""
    csv_content = "x,y,z\n1.0,2.0,128\n3.0,4.0,64\n5.0,6.0,32"
    path = tmp_path / "test_cartesian.csv"
    path.write_text(csv_content)
    return path


@pytest.fixture
def sample_ply(tmp_path: Path) -> Path:
    """Create a minimal valid PLY file."""
    ply_content = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
1.0 2.0 3.0 255 0 0
4.0 5.0 6.0 0 255 0
7.0 8.0 9.0 0 0 255
"""
    path = tmp_path / "test_cloud.ply"
    path.write_text(ply_content)
    return path


@pytest.fixture
def sample_angles():
    """Sample angle array in radians."""
    return np.linspace(0, 2 * np.pi, 100, endpoint=False, dtype=np.float32)


@pytest.fixture
def sample_ranges():
    """Sample range array."""
    return np.linspace(0, 10, 50, dtype=np.float32)[None, :].repeat(100, axis=0)

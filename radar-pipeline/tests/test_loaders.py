"""Tests for data loaders."""

import pytest
import numpy as np
from pathlib import Path

from radar_pipeline.core.loaders import (
    load_radar_csv,
    load_cartesian_csv,
    load_ply,
    detect_csv_format,
)
from radar_pipeline.config import RadarConfig


def test_load_radar_csv(sample_radar_csv: Path):
    """Test loading radar CSV file."""
    config = RadarConfig()
    sweep = load_radar_csv(sample_radar_csv, config)

    assert sweep.angles_rad.shape[0] == 2
    assert sweep.intensities.shape == (2, 1024)
    assert sweep.gain == 75
    assert sweep.source_path == sample_radar_csv


def test_load_cartesian_csv(sample_cartesian_csv: Path):
    """Test loading Cartesian CSV file."""
    cloud = load_cartesian_csv(sample_cartesian_csv)

    assert cloud.size == 3
    np.testing.assert_array_equal(cloud.x, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(cloud.y, [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(cloud.z, [128, 64, 32])


def test_load_ply(sample_ply: Path):
    """Test loading PLY file."""
    cloud = load_ply(sample_ply)

    assert cloud.size == 3
    np.testing.assert_array_equal(cloud.x, [1.0, 4.0, 7.0])
    np.testing.assert_array_equal(cloud.y, [2.0, 5.0, 8.0])
    np.testing.assert_array_equal(cloud.z, [3.0, 6.0, 9.0])
    assert cloud.colors is not None
    assert cloud.colors.shape == (3, 3)


def test_detect_csv_format_radar(sample_radar_csv: Path):
    """Test CSV format detection for radar format."""
    fmt = detect_csv_format(sample_radar_csv)
    assert fmt == "radar"


def test_detect_csv_format_cartesian(sample_cartesian_csv: Path):
    """Test CSV format detection for Cartesian format."""
    fmt = detect_csv_format(sample_cartesian_csv)
    assert fmt == "cartesian"


def test_load_ply_invalid(tmp_path: Path):
    """Test loading invalid PLY file raises error."""
    invalid_ply = tmp_path / "invalid.ply"
    invalid_ply.write_text("not a ply file")

    with pytest.raises(ValueError, match="not a PLY file"):
        load_ply(invalid_ply)

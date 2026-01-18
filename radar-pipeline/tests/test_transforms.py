"""Tests for coordinate transformations."""

import pytest
import numpy as np

from radar_pipeline.core.transforms import (
    polar_to_cartesian,
    subsample_cloud,
    apply_stride,
    intensity_to_colors,
)
from radar_pipeline.core.loaders import PointCloud


def test_polar_to_cartesian_basic():
    """Test basic polar to Cartesian conversion."""
    angles = np.array([0, np.pi / 2, np.pi], dtype=np.float32)
    ranges = np.array([[1], [1], [1]], dtype=np.float32)

    x, y = polar_to_cartesian(angles, ranges)

    np.testing.assert_allclose(x[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(y[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(x[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(y[1], 1.0, atol=1e-6)
    np.testing.assert_allclose(x[2], -1.0, atol=1e-6)
    np.testing.assert_allclose(y[2], 0.0, atol=1e-6)


def test_polar_to_cartesian_multiple_ranges():
    """Test conversion with multiple range bins."""
    angles = np.array([0, np.pi / 2], dtype=np.float32)
    ranges = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)

    x, y = polar_to_cartesian(angles, ranges)

    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    np.testing.assert_allclose(x[0], [1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(y[1], [1, 2, 3], atol=1e-6)


def test_subsample_cloud_no_subsample():
    """Test subsampling when points below limit."""
    cloud = PointCloud(
        x=np.array([1, 2, 3], dtype=np.float32),
        y=np.array([4, 5, 6], dtype=np.float32),
        z=np.array([7, 8, 9], dtype=np.float32),
    )

    result, stride = subsample_cloud(cloud, max_points=10)

    assert result.size == 3
    assert stride == 1


def test_subsample_cloud_with_subsample():
    """Test subsampling when points exceed limit."""
    n = 1000
    cloud = PointCloud(
        x=np.arange(n, dtype=np.float32),
        y=np.arange(n, dtype=np.float32),
        z=np.arange(n, dtype=np.float32),
    )

    result, stride = subsample_cloud(cloud, max_points=100)

    assert result.size == 100
    assert stride == 10


def test_apply_stride():
    """Test stride subsampling."""
    cloud = PointCloud(
        x=np.arange(10, dtype=np.float32),
        y=np.arange(10, dtype=np.float32),
        z=np.arange(10, dtype=np.float32),
    )

    result = apply_stride(cloud, stride=2)

    assert result.size == 5
    np.testing.assert_array_equal(result.x, [0, 2, 4, 6, 8])


def test_apply_stride_no_change():
    """Test stride of 1 returns same cloud."""
    cloud = PointCloud(
        x=np.arange(5, dtype=np.float32),
        y=np.arange(5, dtype=np.float32),
        z=np.arange(5, dtype=np.float32),
    )

    result = apply_stride(cloud, stride=1)

    assert result.size == cloud.size


def test_intensity_to_colors():
    """Test intensity to grayscale conversion."""
    values = np.array([0, 128, 255], dtype=np.float32)

    colors = intensity_to_colors(values)

    assert colors.shape == (3, 3)
    np.testing.assert_array_equal(colors[0], [0, 0, 0])
    np.testing.assert_array_equal(colors[1], [128, 128, 128])
    np.testing.assert_array_equal(colors[2], [255, 255, 255])

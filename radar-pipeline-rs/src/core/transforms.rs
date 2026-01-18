//! Coordinate transformations and point cloud operations.
//!
//! This module provides functions for converting radar sweep data between
//! polar and Cartesian coordinates, filtering, subsampling, and color mapping.
//! All computationally intensive operations are parallelized using Rayon.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::config::{ProcessingConfig, RadarConfig};
use super::loaders::{PointCloud, RadarSweep};

/// Convert polar coordinates to Cartesian coordinates.
///
/// Computes x = ranges * cos(angle) and y = ranges * sin(angle) for each
/// angle/range pair. The computation is parallelized across angles.
///
/// # Arguments
///
/// * `angles_rad` - Angles in radians, one per row
/// * `ranges` - Range values for each angle, shape [num_angles][num_bins]
///
/// # Returns
///
/// Tuple of (x_coords, y_coords), each with shape [num_angles][num_bins]
///
/// # Example
///
/// ```ignore
/// let angles = vec![0.0, std::f32::consts::FRAC_PI_2];
/// let ranges = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
/// let (x, y) = polar_to_cartesian(&angles, &ranges);
/// ```
pub fn polar_to_cartesian(
    angles_rad: &[f32],
    ranges: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    debug_assert_eq!(
        angles_rad.len(),
        ranges.len(),
        "angles and ranges must have same length"
    );

    let results: Vec<(Vec<f32>, Vec<f32>)> = angles_rad
        .par_iter()
        .zip(ranges.par_iter())
        .map(|(&angle, range_row)| {
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            // Pre-allocate output vectors
            let mut x_row = Vec::with_capacity(range_row.len());
            let mut y_row = Vec::with_capacity(range_row.len());

            for &r in range_row {
                x_row.push(r * cos_a);
                y_row.push(r * sin_a);
            }

            (x_row, y_row)
        })
        .collect();

    // Unzip the results
    let mut x_coords = Vec::with_capacity(results.len());
    let mut y_coords = Vec::with_capacity(results.len());

    for (x_row, y_row) in results {
        x_coords.push(x_row);
        y_coords.push(y_row);
    }

    (x_coords, y_coords)
}

/// Convert radar sweep to point cloud with intensity filtering.
///
/// Performs polar-to-Cartesian conversion, applies intensity threshold,
/// and optionally subsamples the resulting points.
///
/// # Arguments
///
/// * `sweep` - Radar sweep data containing angles, ranges, and intensities
/// * `config` - Processing configuration with threshold and stride settings
///
/// # Returns
///
/// A `PointCloud` with x, y coordinates and intensity as z values
pub fn sweep_to_point_cloud(sweep: &RadarSweep, config: &ProcessingConfig) -> PointCloud {
    let num_angles = sweep.angles_rad.len();
    let num_bins = if !sweep.ranges.is_empty() {
        sweep.ranges[0].len()
    } else {
        return PointCloud::empty();
    };

    // Parallel conversion: each angle row is processed independently
    let points_per_row: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = sweep
        .angles_rad
        .par_iter()
        .enumerate()
        .map(|(i, &angle)| {
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let mut x_pts = Vec::new();
            let mut y_pts = Vec::new();
            let mut z_pts = Vec::new();

            let range_row = &sweep.ranges[i];
            let intensity_row = &sweep.intensities[i];

            for j in 0..num_bins {
                let intensity = intensity_row[j];
                if intensity > config.intensity_threshold {
                    let r = range_row[j];
                    x_pts.push(r * cos_a);
                    y_pts.push(r * sin_a);
                    z_pts.push(intensity);
                }
            }

            (x_pts, y_pts, z_pts)
        })
        .collect();

    // Estimate total capacity
    let total_points: usize = points_per_row.iter().map(|(x, _, _)| x.len()).sum();

    // Flatten results
    let mut x_all = Vec::with_capacity(total_points);
    let mut y_all = Vec::with_capacity(total_points);
    let mut z_all = Vec::with_capacity(total_points);

    for (x_row, y_row, z_row) in points_per_row {
        x_all.extend(x_row);
        y_all.extend(y_row);
        z_all.extend(z_row);
    }

    // Apply stride subsampling
    if config.point_stride > 1 {
        let stride = config.point_stride;
        let strided_len = (x_all.len() + stride - 1) / stride;

        let mut x_strided = Vec::with_capacity(strided_len);
        let mut y_strided = Vec::with_capacity(strided_len);
        let mut z_strided = Vec::with_capacity(strided_len);

        for i in (0..x_all.len()).step_by(stride) {
            x_strided.push(x_all[i]);
            y_strided.push(y_all[i]);
            z_strided.push(z_all[i]);
        }

        PointCloud::from_xyz(x_strided, y_strided, z_strided)
    } else {
        PointCloud::from_xyz(x_all, y_all, z_all)
    }
}

/// Convert sweep to points using uniform range bins.
///
/// A simplified conversion that assumes uniform range bin widths rather than
/// per-row scaling. This is faster when range bins are known to be uniform.
///
/// # Arguments
///
/// * `angles_rad` - Angles in radians
/// * `intensities` - Intensity matrix [num_angles][num_bins]
/// * `range_bin_width` - Width of each range bin in meters
/// * `range_start` - Starting range in meters
/// * `min_intensity` - Minimum intensity threshold for filtering
/// * `stride` - Subsampling stride (1 = no subsampling)
///
/// # Returns
///
/// Tuple of (x, y, z) point vectors
pub fn sweep_to_points_simple(
    angles_rad: &[f32],
    intensities: &[Vec<f32>],
    range_bin_width: f32,
    range_start: f32,
    min_intensity: f32,
    stride: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    if angles_rad.is_empty() || intensities.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let num_bins = intensities[0].len();

    // Pre-compute range values for each bin
    let ranges: Vec<f32> = (0..num_bins)
        .map(|i| range_start + (i as f32) * range_bin_width)
        .collect();

    // Parallel processing across angles
    let points_per_row: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = angles_rad
        .par_iter()
        .zip(intensities.par_iter())
        .map(|(&angle, intensity_row)| {
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let mut x_pts = Vec::new();
            let mut y_pts = Vec::new();
            let mut z_pts = Vec::new();

            for (j, &intensity) in intensity_row.iter().enumerate() {
                if intensity > min_intensity {
                    let r = ranges[j];
                    x_pts.push(r * cos_a);
                    y_pts.push(r * sin_a);
                    z_pts.push(intensity);
                }
            }

            (x_pts, y_pts, z_pts)
        })
        .collect();

    // Calculate total points
    let total_points: usize = points_per_row.iter().map(|(x, _, _)| x.len()).sum();

    // Flatten results
    let mut x_all = Vec::with_capacity(total_points);
    let mut y_all = Vec::with_capacity(total_points);
    let mut z_all = Vec::with_capacity(total_points);

    for (x_row, y_row, z_row) in points_per_row {
        x_all.extend(x_row);
        y_all.extend(y_row);
        z_all.extend(z_row);
    }

    // Apply stride subsampling
    if stride > 1 {
        let strided_len = (x_all.len() + stride - 1) / stride;

        let mut x_strided = Vec::with_capacity(strided_len);
        let mut y_strided = Vec::with_capacity(strided_len);
        let mut z_strided = Vec::with_capacity(strided_len);

        for i in (0..x_all.len()).step_by(stride) {
            x_strided.push(x_all[i]);
            y_strided.push(y_all[i]);
            z_strided.push(z_all[i]);
        }

        (x_strided, y_strided, z_strided)
    } else {
        (x_all, y_all, z_all)
    }
}

/// Randomly subsample point cloud to maximum number of points.
///
/// If the cloud has fewer points than `max_points`, returns the original
/// cloud unchanged with stride factor 1.
///
/// # Arguments
///
/// * `cloud` - Input point cloud
/// * `max_points` - Maximum number of points to keep
///
/// # Returns
///
/// Tuple of (subsampled_cloud, stride_factor) where stride_factor is the
/// approximate reduction ratio (ceiling of n/max_points)
pub fn subsample_cloud(cloud: &PointCloud, max_points: usize) -> (PointCloud, usize) {
    let n = cloud.size();

    if n <= max_points {
        return (cloud.clone(), 1);
    }

    // Calculate stride factor
    let stride_factor = (n + max_points - 1) / max_points;

    // Generate random indices using a simple LCG for reproducibility
    // For true randomness, use rand crate
    let mut indices: Vec<usize> = (0..n).collect();

    // Fisher-Yates shuffle (partial - only need max_points elements)
    let mut state: u64 = 12345; // Seed
    for i in 0..max_points.min(n) {
        // Simple LCG: state = (a * state + c) mod m
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = i + ((state as usize) % (n - i));
        indices.swap(i, j);
    }

    indices.truncate(max_points);

    // Sort indices for cache-friendly access
    indices.sort_unstable();

    // Extract subsampled points
    let mut x = Vec::with_capacity(max_points);
    let mut y = Vec::with_capacity(max_points);
    let mut z = Vec::with_capacity(max_points);

    for &idx in &indices {
        x.push(cloud.x[idx]);
        y.push(cloud.y[idx]);
        z.push(cloud.z[idx]);
    }

    let colors = cloud.colors.as_ref().map(|c| {
        indices.iter().map(|&idx| c[idx]).collect()
    });

    (PointCloud { x, y, z, colors }, stride_factor)
}

/// Apply regular stride subsampling to point cloud.
///
/// Keeps every Nth point where N is the stride value.
/// If stride <= 1, returns a clone of the original cloud.
///
/// # Arguments
///
/// * `cloud` - Input point cloud
/// * `stride` - Stride factor (keep every Nth point)
///
/// # Returns
///
/// Subsampled point cloud
pub fn apply_stride(cloud: &PointCloud, stride: usize) -> PointCloud {
    if stride <= 1 {
        return cloud.clone();
    }

    let new_len = (cloud.size() + stride - 1) / stride;

    let mut x = Vec::with_capacity(new_len);
    let mut y = Vec::with_capacity(new_len);
    let mut z = Vec::with_capacity(new_len);

    for i in (0..cloud.size()).step_by(stride) {
        x.push(cloud.x[i]);
        y.push(cloud.y[i]);
        z.push(cloud.z[i]);
    }

    let colors = cloud.colors.as_ref().map(|c| {
        (0..cloud.size())
            .step_by(stride)
            .map(|i| c[i])
            .collect()
    });

    PointCloud { x, y, z, colors }
}

/// Apply vertical offset to point cloud z-values.
///
/// Creates a new point cloud with the offset added to all z values.
/// Useful for stacking point clouds from different gain levels.
///
/// # Arguments
///
/// * `cloud` - Input point cloud
/// * `offset` - Z offset to add to all points
///
/// # Returns
///
/// New point cloud with offset z-values
pub fn apply_z_offset(cloud: &PointCloud, offset: f32) -> PointCloud {
    let z: Vec<f32> = cloud.z.par_iter().map(|&val| val + offset).collect();

    PointCloud {
        x: cloud.x.clone(),
        y: cloud.y.clone(),
        z,
        colors: cloud.colors.clone(),
    }
}

/// Map intensity values to grayscale RGB colors.
///
/// Clamps intensity values to 0-255 range and creates grayscale RGB
/// where R=G=B=intensity. Parallelized using Rayon.
///
/// # Arguments
///
/// * `values` - Intensity values (expected 0-255 scale)
///
/// # Returns
///
/// Vector of RGB color triplets [r, g, b]
pub fn intensity_to_colors(values: &[f32]) -> Vec<[u8; 3]> {
    values
        .par_iter()
        .map(|&v| {
            let clamped = v.clamp(0.0, 255.0) as u8;
            [clamped, clamped, clamped]
        })
        .collect()
}

/// Map all points to a constant gain color.
///
/// Looks up the color for the given gain level and applies it to all points.
/// If the gain is not found in the color map, uses a default gray color.
///
/// # Arguments
///
/// * `values` - Array of values (used only for determining output length)
/// * `gain` - Gain level to look up
/// * `gain_colors` - Mapping of gain levels to RGB colors
///
/// # Returns
///
/// Vector of RGB color triplets, all set to the same gain color
pub fn gain_to_colors(
    values: &[f32],
    gain: i32,
    gain_colors: &HashMap<i32, [u8; 3]>,
) -> Vec<[u8; 3]> {
    let default_color = [180u8, 180, 180];
    let color = gain_colors.get(&gain).copied().unwrap_or(default_color);

    // Parallel creation of color array
    values.par_iter().map(|_| color).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    #[test]
    fn test_polar_to_cartesian_basic() {
        let angles = vec![0.0, FRAC_PI_2, PI];
        let ranges = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];

        let (x, y) = polar_to_cartesian(&angles, &ranges);

        // At angle 0: x = r, y = 0
        assert!((x[0][0] - 1.0).abs() < 1e-6);
        assert!((y[0][0]).abs() < 1e-6);

        // At angle PI/2: x = 0, y = r
        assert!((x[1][0]).abs() < 1e-6);
        assert!((y[1][0] - 1.0).abs() < 1e-6);

        // At angle PI: x = -r, y = 0
        assert!((x[2][0] + 1.0).abs() < 1e-6);
        assert!((y[2][0]).abs() < 1e-6);
    }

    #[test]
    fn test_polar_to_cartesian_empty() {
        let angles: Vec<f32> = vec![];
        let ranges: Vec<Vec<f32>> = vec![];

        let (x, y) = polar_to_cartesian(&angles, &ranges);

        assert!(x.is_empty());
        assert!(y.is_empty());
    }

    #[test]
    fn test_sweep_to_points_simple_filtering() {
        let angles = vec![0.0, FRAC_PI_2];
        let intensities = vec![
            vec![10.0, 50.0, 100.0],
            vec![5.0, 60.0, 200.0],
        ];

        let (x, y, z) = sweep_to_points_simple(
            &angles,
            &intensities,
            1.0,  // range_bin_width
            0.0,  // range_start
            30.0, // min_intensity threshold
            1,    // no stride
        );

        // Should filter out intensities <= 30
        // Remaining: 50, 100 from row 0; 60, 200 from row 1
        assert_eq!(z.len(), 4);
        assert!(z.contains(&50.0));
        assert!(z.contains(&100.0));
        assert!(z.contains(&60.0));
        assert!(z.contains(&200.0));
    }

    #[test]
    fn test_sweep_to_points_simple_with_stride() {
        let angles = vec![0.0];
        let intensities = vec![vec![100.0; 10]]; // 10 points above threshold

        let (x, y, z) = sweep_to_points_simple(
            &angles,
            &intensities,
            1.0,
            0.0,
            0.0, // no filtering
            3,   // stride of 3
        );

        // 10 points with stride 3: indices 0, 3, 6, 9 = 4 points
        assert_eq!(z.len(), 4);
    }

    #[test]
    fn test_apply_stride() {
        let cloud = PointCloud {
            x: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            y: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            z: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            colors: None,
        };

        let strided = apply_stride(&cloud, 2);

        assert_eq!(strided.x, vec![1.0, 3.0, 5.0]);
        assert_eq!(strided.y, vec![1.0, 3.0, 5.0]);
        assert_eq!(strided.z, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_apply_stride_no_change() {
        let cloud = PointCloud {
            x: vec![1.0, 2.0, 3.0],
            y: vec![1.0, 2.0, 3.0],
            z: vec![1.0, 2.0, 3.0],
            colors: None,
        };

        let strided = apply_stride(&cloud, 1);

        assert_eq!(strided.size(), 3);
    }

    #[test]
    fn test_apply_z_offset() {
        let cloud = PointCloud {
            x: vec![1.0, 2.0],
            y: vec![3.0, 4.0],
            z: vec![5.0, 6.0],
            colors: None,
        };

        let offset_cloud = apply_z_offset(&cloud, 100.0);

        assert_eq!(offset_cloud.z, vec![105.0, 106.0]);
        assert_eq!(offset_cloud.x, cloud.x);
        assert_eq!(offset_cloud.y, cloud.y);
    }

    #[test]
    fn test_intensity_to_colors() {
        let values = vec![0.0, 127.5, 255.0, 300.0, -10.0];

        let colors = intensity_to_colors(&values);

        assert_eq!(colors[0], [0, 0, 0]);       // 0 -> black
        assert_eq!(colors[1], [127, 127, 127]); // 127.5 truncated to 127
        assert_eq!(colors[2], [255, 255, 255]); // 255 -> white
        assert_eq!(colors[3], [255, 255, 255]); // 300 clamped to 255
        assert_eq!(colors[4], [0, 0, 0]);       // -10 clamped to 0
    }

    #[test]
    fn test_gain_to_colors() {
        let mut gain_colors = HashMap::new();
        gain_colors.insert(40, [0, 114, 255]);
        gain_colors.insert(50, [0, 200, 83]);

        let values = vec![1.0, 2.0, 3.0];

        let colors = gain_to_colors(&values, 40, &gain_colors);

        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], [0, 114, 255]);
        assert_eq!(colors[1], [0, 114, 255]);
        assert_eq!(colors[2], [0, 114, 255]);
    }

    #[test]
    fn test_gain_to_colors_default() {
        let gain_colors = HashMap::new();
        let values = vec![1.0, 2.0];

        let colors = gain_to_colors(&values, 99, &gain_colors);

        // Should use default gray
        assert_eq!(colors[0], [180, 180, 180]);
        assert_eq!(colors[1], [180, 180, 180]);
    }

    #[test]
    fn test_subsample_cloud_no_reduction() {
        let cloud = PointCloud {
            x: vec![1.0, 2.0, 3.0],
            y: vec![1.0, 2.0, 3.0],
            z: vec![1.0, 2.0, 3.0],
            colors: None,
        };

        let (result, stride) = subsample_cloud(&cloud, 10);

        assert_eq!(result.size(), 3);
        assert_eq!(stride, 1);
    }

    #[test]
    fn test_subsample_cloud_with_reduction() {
        let cloud = PointCloud {
            x: (0..100).map(|i| i as f32).collect(),
            y: (0..100).map(|i| i as f32).collect(),
            z: (0..100).map(|i| i as f32).collect(),
            colors: None,
        };

        let (result, stride) = subsample_cloud(&cloud, 10);

        assert_eq!(result.size(), 10);
        assert_eq!(stride, 10); // 100 / 10 = 10
    }
}

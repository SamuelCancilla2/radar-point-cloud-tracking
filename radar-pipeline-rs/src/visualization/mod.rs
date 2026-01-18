//! Visualization tools for point cloud data.
//!
//! This module provides functions to generate 2D scatter plot visualizations
//! of point clouds using the plotters library.

use std::path::Path;

use plotters::prelude::*;
use plotters_bitmap::BitMapBackend;
use thiserror::Error;

use crate::core::loaders::PointCloud;

/// Errors that can occur during visualization.
#[derive(Error, Debug)]
pub enum VisualizationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Plotting error: {0}")]
    PlottingError(String),

    #[error("Empty point cloud")]
    EmptyPointCloud,
}

/// Result type for visualization operations.
pub type Result<T> = std::result::Result<T, VisualizationError>;

/// Default plot width in pixels.
const DEFAULT_WIDTH: u32 = 1920;

/// Default plot height in pixels.
const DEFAULT_HEIGHT: u32 = 1080;

/// Color palette for cluster visualization.
const CLUSTER_COLORS: &[(u8, u8, u8)] = &[
    (228, 26, 28),   // Red
    (55, 126, 184),  // Blue
    (77, 175, 74),   // Green
    (152, 78, 163),  // Purple
    (255, 127, 0),   // Orange
    (255, 255, 51),  // Yellow
    (166, 86, 40),   // Brown
    (247, 129, 191), // Pink
    (153, 153, 153), // Gray
    (0, 206, 209),   // Turquoise
    (138, 43, 226),  // Blue Violet
    (50, 205, 50),   // Lime Green
    (255, 20, 147),  // Deep Pink
    (0, 191, 255),   // Deep Sky Blue
    (255, 215, 0),   // Gold
];

/// Noise color (gray) for unclustered points (label = -1).
const NOISE_COLOR: (u8, u8, u8) = (128, 128, 128);

/// Plot a 2D scatter plot (x vs y) of a point cloud and save as PNG.
///
/// # Arguments
///
/// * `output_path` - Path to save the PNG image
/// * `cloud` - The point cloud to visualize
/// * `_title` - Title for the plot (unused - no fonts on WSL)
/// * `max_points` - Maximum number of points to plot (subsamples if exceeded)
/// * `alpha` - Alpha/transparency value for points (0.0 to 1.0)
pub fn plot_point_cloud(
    output_path: &Path,
    cloud: &PointCloud,
    _title: &str,
    max_points: usize,
    alpha: f32,
) -> Result<()> {
    if cloud.is_empty() {
        return Err(VisualizationError::EmptyPointCloud);
    }

    let n = cloud.len();

    // Compute subsampling step
    let step = if n > max_points { n / max_points } else { 1 };
    let num_points_to_plot = if n > max_points { max_points } else { n };

    // Collect points to plot with optional subsampling
    let mut points: Vec<(f32, f32, RGBAColor)> = Vec::with_capacity(num_points_to_plot);

    let alpha_f64 = (alpha.clamp(0.0, 1.0)) as f64;

    for i in (0..n).step_by(step) {
        let x = cloud.x[i];
        let y = cloud.y[i];

        let color = if let Some(ref colors) = cloud.colors {
            let c = colors[i];
            RGBAColor(c[0], c[1], c[2], alpha_f64)
        } else {
            RGBAColor(100, 149, 237, alpha_f64) // Cornflower blue default
        };

        points.push((x, y, color));
    }

    // Compute bounds with padding
    let (x_min, x_max, y_min, y_max) = compute_bounds(&points);
    let x_padding = (x_max - x_min) * 0.05;
    let y_padding = (y_max - y_min) * 0.05;

    // Create the plot
    let root = BitMapBackend::new(output_path, (DEFAULT_WIDTH, DEFAULT_HEIGHT))
        .into_drawing_area();

    root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(
            (x_min - x_padding)..(x_max + x_padding),
            (y_min - y_padding)..(y_max + y_padding),
        )
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    // Draw points
    chart
        .draw_series(points.iter().map(|(x, y, color)| {
            Circle::new((*x, *y), 2, color.filled())
        }))
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    Ok(())
}

/// Plot a 2D scatter plot of a labeled point cloud with colors by cluster.
pub fn plot_labeled_cloud(
    output_path: &Path,
    coords: &[[f32; 3]],
    labels: &[i32],
    _title: &str,
    max_points: usize,
) -> Result<()> {
    if coords.is_empty() {
        return Err(VisualizationError::EmptyPointCloud);
    }

    let n = coords.len();

    // Compute subsampling step
    let step = if n > max_points { n / max_points } else { 1 };
    let num_points_to_plot = if n > max_points { max_points } else { n };

    // Collect points with colors based on labels
    let mut points: Vec<(f32, f32, RGBColor)> = Vec::with_capacity(num_points_to_plot);

    for i in (0..n).step_by(step) {
        let x = coords[i][0];
        let y = coords[i][1];
        let label = labels[i];

        let color = if label < 0 {
            RGBColor(NOISE_COLOR.0, NOISE_COLOR.1, NOISE_COLOR.2)
        } else {
            let color_idx = (label as usize) % CLUSTER_COLORS.len();
            let c = CLUSTER_COLORS[color_idx];
            RGBColor(c.0, c.1, c.2)
        };

        points.push((x, y, color));
    }

    // Convert for bounds computation
    let points_for_bounds: Vec<(f32, f32, RGBAColor)> = points
        .iter()
        .map(|(x, y, c)| (*x, *y, RGBAColor(c.0, c.1, c.2, 1.0)))
        .collect();

    let (x_min, x_max, y_min, y_max) = compute_bounds(&points_for_bounds);
    let x_padding = (x_max - x_min) * 0.05;
    let y_padding = (y_max - y_min) * 0.05;

    // Create the plot
    let root = BitMapBackend::new(output_path, (DEFAULT_WIDTH, DEFAULT_HEIGHT))
        .into_drawing_area();

    root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(
            (x_min - x_padding)..(x_max + x_padding),
            (y_min - y_padding)..(y_max + y_padding),
        )
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    // Draw points
    chart
        .draw_series(points.iter().map(|(x, y, color)| {
            Circle::new((*x, *y), 2, color.filled())
        }))
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

    Ok(())
}

/// Compute the bounds (min/max) for x and y coordinates.
fn compute_bounds(points: &[(f32, f32, RGBAColor)]) -> (f32, f32, f32, f32) {
    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;

    for (x, y, _) in points {
        if *x < x_min { x_min = *x; }
        if *x > x_max { x_max = *x; }
        if *y < y_min { y_min = *y; }
        if *y > y_max { y_max = *y; }
    }

    if (x_max - x_min).abs() < f32::EPSILON {
        x_min -= 1.0;
        x_max += 1.0;
    }
    if (y_max - y_min).abs() < f32::EPSILON {
        y_min -= 1.0;
        y_max += 1.0;
    }

    (x_min, x_max, y_min, y_max)
}

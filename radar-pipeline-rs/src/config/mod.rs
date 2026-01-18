//! Configuration types for the radar pipeline.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for radar gain levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GainConfig {
    /// Gain values to process (e.g., [40, 50, 75])
    #[serde(default = "default_gain_values")]
    pub values: Vec<i32>,

    /// RGB colors for each gain level
    #[serde(default = "default_gain_colors")]
    pub colors: HashMap<i32, [u8; 3]>,

    /// Z-axis offsets for stacking by gain
    #[serde(default = "default_z_offsets")]
    pub z_offsets: HashMap<i32, f32>,
}

fn default_gain_values() -> Vec<i32> {
    vec![40, 50, 75]
}

fn default_gain_colors() -> HashMap<i32, [u8; 3]> {
    let mut colors = HashMap::new();
    colors.insert(40, [0, 114, 255]); // blue
    colors.insert(50, [0, 200, 83]); // green
    colors.insert(75, [255, 87, 34]); // orange
    colors
}

fn default_z_offsets() -> HashMap<i32, f32> {
    let mut offsets = HashMap::new();
    offsets.insert(75, 0.0); // bottom layer
    offsets.insert(50, 250.0); // middle layer
    offsets.insert(40, 500.0); // top layer
    offsets
}

impl Default for GainConfig {
    fn default() -> Self {
        Self {
            values: default_gain_values(),
            colors: default_gain_colors(),
            z_offsets: default_z_offsets(),
        }
    }
}

/// Configuration for radar data parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarConfig {
    /// Scale factor for angle conversion
    #[serde(default = "default_angle_scale")]
    pub angle_scale: f32,

    /// Number of echo/intensity columns in radar CSV
    #[serde(default = "default_num_echo_columns")]
    pub num_echo_columns: usize,

    /// Width of each range bin in meters
    #[serde(default = "default_range_bin_width")]
    pub range_bin_width_m: f32,

    /// Starting range in meters
    #[serde(default)]
    pub range_start_m: f32,
}

fn default_angle_scale() -> f32 {
    360.0 / 8196.0
}

fn default_num_echo_columns() -> usize {
    1024
}

fn default_range_bin_width() -> f32 {
    0.5
}

impl Default for RadarConfig {
    fn default() -> Self {
        Self {
            angle_scale: default_angle_scale(),
            num_echo_columns: default_num_echo_columns(),
            range_bin_width_m: default_range_bin_width(),
            range_start_m: 0.0,
        }
    }
}

/// Configuration for point cloud processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Minimum intensity threshold for including points
    #[serde(default)]
    pub intensity_threshold: f32,

    /// Stride for subsampling points
    #[serde(default = "default_point_stride")]
    pub point_stride: usize,

    /// Maximum points per gain level
    #[serde(default = "default_max_points_per_gain")]
    pub max_points_per_gain: usize,

    /// Maximum points in stacked cloud
    #[serde(default = "default_max_points_stack")]
    pub max_points_stack: usize,

    /// Maximum points for plotting
    #[serde(default = "default_plot_max_points")]
    pub plot_max_points: usize,
}

fn default_point_stride() -> usize {
    16
}

fn default_max_points_per_gain() -> usize {
    10_000_000
}

fn default_max_points_stack() -> usize {
    20_000_000
}

fn default_plot_max_points() -> usize {
    1_000_000
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            intensity_threshold: 0.0,
            point_stride: default_point_stride(),
            max_points_per_gain: default_max_points_per_gain(),
            max_points_stack: default_max_points_stack(),
            plot_max_points: default_plot_max_points(),
        }
    }
}

/// Configuration for ST-DBSCAN clustering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Spatial epsilon (neighborhood radius)
    #[serde(default = "default_eps_space")]
    pub eps_space: f32,

    /// Temporal epsilon (time difference threshold)
    #[serde(default = "default_eps_time")]
    pub eps_time: f32,

    /// Minimum samples to form a cluster
    #[serde(default = "default_min_samples")]
    pub min_samples: usize,

    /// Maximum points to process
    #[serde(default = "default_clustering_max_points")]
    pub max_points: usize,
}

fn default_eps_space() -> f32 {
    5.0
}

fn default_eps_time() -> f32 {
    1.0
}

fn default_min_samples() -> usize {
    10
}

fn default_clustering_max_points() -> usize {
    10_000_000
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            eps_space: default_eps_space(),
            eps_time: default_eps_time(),
            min_samples: default_min_samples(),
            max_points: default_clustering_max_points(),
        }
    }
}

/// Main pipeline configuration combining all sub-configs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub gains: GainConfig,

    #[serde(default)]
    pub radar: RadarConfig,

    #[serde(default)]
    pub processing: ProcessingConfig,

    #[serde(default)]
    pub clustering: ClusteringConfig,
}

impl PipelineConfig {
    /// Load configuration from a YAML file.
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: PipelineConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a YAML file.
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_gain_config() {
        let config = GainConfig::default();
        assert_eq!(config.values, vec![40, 50, 75]);
        assert_eq!(config.colors.get(&40), Some(&[0, 114, 255]));
    }

    #[test]
    fn test_default_pipeline_config() {
        let config = PipelineConfig::default();
        assert_eq!(config.radar.num_echo_columns, 1024);
        assert_eq!(config.clustering.min_samples, 10);
    }
}

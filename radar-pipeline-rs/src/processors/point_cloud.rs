//! Point cloud building and stacking operations.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use rayon::prelude::*;
use regex::Regex;
use thiserror::Error;

use crate::config::{GainConfig, ProcessingConfig, RadarConfig};

/// Container for 3D point cloud data.
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// X coordinates
    pub x: Vec<f32>,
    /// Y coordinates
    pub y: Vec<f32>,
    /// Z coordinates (or intensity values)
    pub z: Vec<f32>,
    /// Optional RGB colors for each point
    pub colors: Option<Vec<[u8; 3]>>,
}

impl PointCloud {
    /// Return number of points in the cloud.
    pub fn size(&self) -> usize {
        self.x.len()
    }

    /// Create an empty point cloud.
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            colors: None,
        }
    }

    /// Create a point cloud with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
            colors: None,
        }
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during point cloud operations.
#[derive(Debug, Error)]
pub enum PointCloudError {
    #[error("No gain CSVs found in directory: {0}")]
    NoGainCsvsFound(PathBuf),

    #[error("Failed to read CSV file: {path}")]
    CsvReadError { path: PathBuf },

    #[error("Failed to write PLY file: {path}")]
    PlyWriteError { path: PathBuf },

    #[error("Invalid CSV format: {0}")]
    InvalidFormat(String),
}

/// Discover gain-specific sweep CSVs in a directory.
///
/// Searches for CSV files with "gain_N" or "gain-N" in their filename
/// and returns a mapping of gain values to file paths.
///
/// # Arguments
///
/// * `directory` - Directory to search for CSV files
///
/// # Returns
///
/// A HashMap mapping gain values to their corresponding file paths.
pub fn find_gain_sweeps(directory: &Path) -> HashMap<i32, PathBuf> {
    let gain_pattern = Regex::new(r"(?i)gain[_-]?(\d+)").unwrap();
    let mut sweeps: HashMap<i32, PathBuf> = HashMap::new();

    let mut csv_files: Vec<PathBuf> = fs::read_dir(directory)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .map(|ext| ext.eq_ignore_ascii_case("csv"))
                .unwrap_or(false)
        })
        .collect();

    csv_files.sort();

    for path in csv_files {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();

        if let Some(captures) = gain_pattern.captures(stem) {
            if let Some(gain_match) = captures.get(1) {
                if let Ok(gain) = gain_match.as_str().parse::<i32>() {
                    sweeps.insert(gain, path);
                }
            }
        }
    }

    sweeps
}

/// Detect whether a CSV is radar sweep format or Cartesian point format.
fn detect_csv_format(path: &Path) -> Result<&'static str> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    if let Some(Ok(header)) = reader.lines().next() {
        let lower_header = header.to_lowercase();
        if lower_header.contains("x") && lower_header.contains("y") && lower_header.contains("z") {
            return Ok("cartesian");
        }
    }

    Ok("radar")
}

/// Load points from a Cartesian CSV file.
fn load_cartesian_csv(path: &Path) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(64 * 1024, file);
    let mut lines = reader.lines();

    // Skip header
    lines.next();

    // Pre-allocate with estimated capacity
    let mut x_pts = Vec::with_capacity(100_000);
    let mut y_pts = Vec::with_capacity(100_000);
    let mut z_pts = Vec::with_capacity(100_000);

    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();

        if fields.len() >= 3 {
            if let (Ok(x), Ok(y), Ok(z)) = (
                fields[0].trim().parse::<f32>(),
                fields[1].trim().parse::<f32>(),
                fields[2].trim().parse::<f32>(),
            ) {
                x_pts.push(x);
                y_pts.push(y);
                z_pts.push(z);
            }
        }
    }

    Ok((x_pts, y_pts, z_pts))
}

/// Load points from a radar sweep CSV file.
fn load_radar_sweep_points(
    path: &Path,
    config: &ProcessingConfig,
    radar_config: &RadarConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(64 * 1024, file);
    let mut lines = reader.lines();

    // Skip header
    lines.next();

    // Collect rows
    let rows: Vec<String> = lines.filter_map(|l| l.ok()).collect();

    if rows.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let num_angles = rows.len();
    let angle_step = 2.0 * std::f32::consts::PI / num_angles as f32;

    // Pre-allocate with estimated capacity
    let estimated_points = num_angles * radar_config.num_echo_columns;
    let mut x_pts = Vec::with_capacity(estimated_points);
    let mut y_pts = Vec::with_capacity(estimated_points);
    let mut z_pts = Vec::with_capacity(estimated_points);

    let threshold = config.intensity_threshold;
    let stride = config.point_stride.max(1);

    for (angle_idx, row) in rows.iter().enumerate() {
        let fields: Vec<&str> = row.split(',').collect();

        if fields.len() < 6 {
            continue;
        }

        let num_bins = fields.len().saturating_sub(5).min(radar_config.num_echo_columns);
        if num_bins == 0 {
            continue;
        }

        let angle_rad = angle_idx as f32 * angle_step;
        let cos_angle = angle_rad.cos();
        let sin_angle = angle_rad.sin();

        // Process echo columns with stride
        for (bin_idx, &field) in fields.iter().skip(5).take(num_bins).enumerate() {
            if bin_idx % stride != 0 {
                continue;
            }

            let intensity: f32 = field.trim().parse().unwrap_or(0.0);

            if intensity > threshold {
                let range =
                    radar_config.range_start_m + bin_idx as f32 * radar_config.range_bin_width_m;
                let x = range * cos_angle;
                let y = range * sin_angle;

                x_pts.push(x);
                y_pts.push(y);
                z_pts.push(intensity);
            }
        }
    }

    Ok((x_pts, y_pts, z_pts))
}

/// Load points from CSV, auto-detecting format.
///
/// Automatically detects whether the CSV is in Cartesian format (x,y,z columns)
/// or radar sweep format, and loads accordingly.
///
/// # Arguments
///
/// * `path` - Path to the CSV file
/// * `config` - Processing configuration
/// * `radar_config` - Radar configuration
///
/// # Returns
///
/// A tuple of (x, y, z) coordinate vectors.
pub fn load_points_from_csv(
    path: &Path,
    config: &ProcessingConfig,
    radar_config: &RadarConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let format = detect_csv_format(path)?;

    match format {
        "cartesian" => load_cartesian_csv(path),
        _ => load_radar_sweep_points(path, config, radar_config),
    }
}

/// Apply gain-specific colors to points.
///
/// Creates an RGB color array with the color associated with the given gain value.
///
/// # Arguments
///
/// * `z` - Z values (intensity), used for sizing the output
/// * `gain` - Gain value
/// * `gain_config` - Gain configuration with color mappings
///
/// # Returns
///
/// A vector of RGB color arrays.
pub fn apply_gain_colors(z: &[f32], gain: i32, gain_config: &GainConfig) -> Vec<[u8; 3]> {
    let default_color = [180u8, 180u8, 180u8];
    let color = gain_config.colors.get(&gain).copied().unwrap_or(default_color);

    vec![color; z.len()]
}

/// Combine multiple point clouds with optional Z offsets.
///
/// Merges multiple gain-specific point clouds into a single combined cloud,
/// optionally applying Z-axis offsets to separate the layers visually.
///
/// # Arguments
///
/// * `clouds` - Slice of (gain, PointCloud) tuples
/// * `apply_offsets` - Whether to apply gain-specific Z offsets
/// * `gain_config` - Gain configuration with offset values
///
/// # Returns
///
/// A combined PointCloud.
pub fn combine_clouds(
    clouds: &[(i32, PointCloud)],
    apply_offsets: bool,
    gain_config: &GainConfig,
) -> PointCloud {
    // Calculate total size for pre-allocation
    let total_size: usize = clouds.iter().map(|(_, c)| c.size()).sum();

    let mut all_x = Vec::with_capacity(total_size);
    let mut all_y = Vec::with_capacity(total_size);
    let mut all_z = Vec::with_capacity(total_size);
    let mut all_colors = Vec::with_capacity(total_size);

    for (gain, cloud) in clouds {
        all_x.extend_from_slice(&cloud.x);
        all_y.extend_from_slice(&cloud.y);

        if apply_offsets {
            let offset = gain_config.z_offsets.get(gain).copied().unwrap_or(0.0);
            all_z.extend(cloud.z.iter().map(|&z| z + offset));
        } else {
            all_z.extend_from_slice(&cloud.z);
        }

        if let Some(ref colors) = cloud.colors {
            all_colors.extend_from_slice(colors);
        } else {
            all_colors.extend(apply_gain_colors(&cloud.z, *gain, gain_config));
        }
    }

    PointCloud {
        x: all_x,
        y: all_y,
        z: all_z,
        colors: Some(all_colors),
    }
}

/// Write a point cloud to PLY format.
fn write_ply(path: &Path, cloud: &PointCloud) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);

    let num_points = cloud.size();

    // Write header
    writeln!(writer, "ply")?;
    writeln!(writer, "format ascii 1.0")?;
    writeln!(writer, "element vertex {}", num_points)?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;
    writeln!(writer, "property uchar red")?;
    writeln!(writer, "property uchar green")?;
    writeln!(writer, "property uchar blue")?;
    writeln!(writer, "end_header")?;

    // Write points
    let default_color = [180u8, 180u8, 180u8];
    for i in 0..num_points {
        let color = cloud
            .colors
            .as_ref()
            .map(|c| c[i])
            .unwrap_or(default_color);
        writeln!(
            writer,
            "{:.6} {:.6} {:.6} {} {} {}",
            cloud.x[i], cloud.y[i], cloud.z[i], color[0], color[1], color[2]
        )?;
    }

    writer.flush()?;
    Ok(())
}

/// Apply stride subsampling to a point cloud.
fn apply_stride(cloud: PointCloud, stride: usize) -> PointCloud {
    if stride <= 1 {
        return cloud;
    }

    let new_x: Vec<f32> = cloud.x.iter().step_by(stride).copied().collect();
    let new_y: Vec<f32> = cloud.y.iter().step_by(stride).copied().collect();
    let new_z: Vec<f32> = cloud.z.iter().step_by(stride).copied().collect();
    let new_colors = cloud
        .colors
        .map(|c| c.iter().step_by(stride).copied().collect());

    PointCloud {
        x: new_x,
        y: new_y,
        z: new_z,
        colors: new_colors,
    }
}

/// Build stacked point clouds from gain-specific CSVs.
///
/// Loads CSV files from the sweep directory, builds individual point clouds
/// for each gain level, and combines them into stacked output files.
///
/// # Arguments
///
/// * `sweep_dir` - Directory containing gain CSVs
/// * `output_dir` - Output directory for PLY files
/// * `config` - Processing configuration
/// * `gain_config` - Gain configuration
/// * `radar_config` - Radar configuration
/// * `generate_flat` - Generate flat (no offset) stack
/// * `generate_offset` - Generate offset (separated layers) stack
/// * `name_prefix` - Output file name prefix
///
/// # Returns
///
/// A HashMap mapping variant names ("flat", "offset") to output paths.
pub fn build_stacked_clouds(
    sweep_dir: &Path,
    output_dir: &Path,
    config: &ProcessingConfig,
    gain_config: &GainConfig,
    radar_config: &RadarConfig,
    generate_flat: bool,
    generate_offset: bool,
    name_prefix: &str,
) -> Result<HashMap<String, PathBuf>> {
    let sweep_files = find_gain_sweeps(sweep_dir);

    if sweep_files.is_empty() {
        return Err(PointCloudError::NoGainCsvsFound(sweep_dir.to_path_buf()).into());
    }

    // Load points in parallel using rayon
    let mut gains: Vec<i32> = sweep_files.keys().copied().collect();
    gains.sort();

    let clouds: Vec<(i32, PointCloud)> = gains
        .par_iter()
        .filter_map(|&gain| {
            let sweep_path = sweep_files.get(&gain)?;

            let (x, y, z) = match load_points_from_csv(sweep_path, config, radar_config) {
                Ok(pts) => pts,
                Err(e) => {
                    eprintln!("Failed to load {}: {}", sweep_path.display(), e);
                    return None;
                }
            };

            let base_points = x.len();

            // Auto-raise stride if file is very large
            let gain_stride =
                config.point_stride.max((base_points / config.max_points_per_gain).max(1));

            let (x, y, z) = if gain_stride > 1 {
                (
                    x.iter().step_by(gain_stride).copied().collect(),
                    y.iter().step_by(gain_stride).copied().collect(),
                    z.iter().step_by(gain_stride).copied().collect(),
                )
            } else {
                (x, y, z)
            };

            let colors = apply_gain_colors(&z, gain, gain_config);

            println!("gain {}: {} points (stride={})", gain, x.len(), gain_stride);

            Some((
                gain,
                PointCloud {
                    x,
                    y,
                    z,
                    colors: Some(colors),
                },
            ))
        })
        .collect();

    // Sort clouds by gain for consistent output
    let mut clouds = clouds;
    clouds.sort_by_key(|(g, _)| *g);

    fs::create_dir_all(output_dir)?;

    let mut outputs = HashMap::new();

    if generate_offset {
        let offset_cloud = combine_clouds(&clouds, true, gain_config);

        // Apply stack stride
        let stack_stride = (offset_cloud.size() / config.max_points_stack).max(1);
        let offset_cloud = if stack_stride > 1 {
            apply_stride(offset_cloud, stack_stride)
        } else {
            offset_cloud
        };

        let offset_path = output_dir.join(format!("{}_v3.ply", name_prefix));
        write_ply(&offset_path, &offset_cloud)?;
        outputs.insert("offset".to_string(), offset_path.clone());
        println!(
            "Offset stack: {} points -> {}",
            offset_cloud.size(),
            offset_path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    if generate_flat {
        let flat_cloud = combine_clouds(&clouds, false, gain_config);

        // Apply stack stride
        let stack_stride = (flat_cloud.size() / config.max_points_stack).max(1);
        let flat_cloud = if stack_stride > 1 {
            apply_stride(flat_cloud, stack_stride)
        } else {
            flat_cloud
        };

        let flat_path = output_dir.join(format!("{}_flat_v3.ply", name_prefix));
        write_ply(&flat_path, &flat_cloud)?;
        outputs.insert("flat".to_string(), flat_path.clone());
        println!(
            "Flat stack: {} points -> {}",
            flat_cloud.size(),
            flat_path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;
    use tempfile::TempDir;

    fn create_gain_csv(dir: &Path, gain: i32, num_rows: usize) -> PathBuf {
        let name = format!("sweep_gain_{}.csv", gain);
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();

        writeln!(file, "Status,Scale,Range,Gain,Angle,Echo_0,Echo_1,Echo_2").unwrap();
        for i in 0..num_rows {
            writeln!(
                file,
                "0,100,50,{},{},{},{},{}",
                gain,
                i * 45 % 360,
                50 + i,
                60 + i,
                70 + i
            )
            .unwrap();
        }
        path
    }

    fn create_cartesian_csv(dir: &Path, name: &str, num_points: usize) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();

        writeln!(file, "x,y,z").unwrap();
        for i in 0..num_points {
            writeln!(file, "{},{},{}", i as f32, (i * 2) as f32, (i * 3) as f32).unwrap();
        }
        path
    }

    #[test]
    fn test_find_gain_sweeps() {
        let temp_dir = TempDir::new().unwrap();

        create_gain_csv(temp_dir.path(), 40, 5);
        create_gain_csv(temp_dir.path(), 50, 5);
        create_gain_csv(temp_dir.path(), 75, 5);

        // Create a non-gain CSV to verify it's ignored
        File::create(temp_dir.path().join("other.csv")).unwrap();

        let sweeps = find_gain_sweeps(temp_dir.path());

        assert_eq!(sweeps.len(), 3);
        assert!(sweeps.contains_key(&40));
        assert!(sweeps.contains_key(&50));
        assert!(sweeps.contains_key(&75));
    }

    #[test]
    fn test_find_gain_sweeps_various_formats() {
        let temp_dir = TempDir::new().unwrap();

        // Test various naming conventions
        File::create(temp_dir.path().join("gain_40.csv")).unwrap();
        File::create(temp_dir.path().join("gain-50.csv")).unwrap();
        File::create(temp_dir.path().join("GAIN_75.csv")).unwrap();
        File::create(temp_dir.path().join("data_gain100.csv")).unwrap();

        let sweeps = find_gain_sweeps(temp_dir.path());

        assert!(sweeps.contains_key(&40));
        assert!(sweeps.contains_key(&50));
        assert!(sweeps.contains_key(&75));
        assert!(sweeps.contains_key(&100));
    }

    #[test]
    fn test_load_points_from_cartesian_csv() {
        let temp_dir = TempDir::new().unwrap();
        let path = create_cartesian_csv(temp_dir.path(), "points.csv", 100);

        let config = ProcessingConfig::default();
        let radar_config = RadarConfig::default();

        let (x, y, z) = load_points_from_csv(&path, &config, &radar_config).unwrap();

        assert_eq!(x.len(), 100);
        assert_eq!(y.len(), 100);
        assert_eq!(z.len(), 100);
        assert_eq!(x[0], 0.0);
        assert_eq!(y[0], 0.0);
        assert_eq!(z[0], 0.0);
    }

    #[test]
    fn test_load_points_from_radar_csv() {
        let temp_dir = TempDir::new().unwrap();
        let path = create_gain_csv(temp_dir.path(), 40, 10);

        let config = ProcessingConfig::default();
        let radar_config = RadarConfig::default();

        let (x, y, z) = load_points_from_csv(&path, &config, &radar_config).unwrap();

        assert!(!x.is_empty());
        assert_eq!(x.len(), y.len());
        assert_eq!(y.len(), z.len());
    }

    #[test]
    fn test_apply_gain_colors() {
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gain_config = GainConfig::default();

        let colors = apply_gain_colors(&z, 40, &gain_config);

        assert_eq!(colors.len(), 5);
        // Gain 40 should be blue
        assert_eq!(colors[0], [0, 114, 255]);
    }

    #[test]
    fn test_apply_gain_colors_unknown_gain() {
        let z = vec![1.0, 2.0, 3.0];
        let gain_config = GainConfig::default();

        let colors = apply_gain_colors(&z, 999, &gain_config);

        assert_eq!(colors.len(), 3);
        // Unknown gain should be gray
        assert_eq!(colors[0], [180, 180, 180]);
    }

    #[test]
    fn test_combine_clouds_no_offset() {
        let gain_config = GainConfig::default();

        let cloud1 = PointCloud {
            x: vec![1.0, 2.0],
            y: vec![3.0, 4.0],
            z: vec![5.0, 6.0],
            colors: Some(vec![[255, 0, 0], [255, 0, 0]]),
        };

        let cloud2 = PointCloud {
            x: vec![7.0, 8.0],
            y: vec![9.0, 10.0],
            z: vec![11.0, 12.0],
            colors: Some(vec![[0, 255, 0], [0, 255, 0]]),
        };

        let clouds = vec![(40, cloud1), (50, cloud2)];
        let combined = combine_clouds(&clouds, false, &gain_config);

        assert_eq!(combined.x.len(), 4);
        assert_eq!(combined.x, vec![1.0, 2.0, 7.0, 8.0]);
        assert_eq!(combined.z, vec![5.0, 6.0, 11.0, 12.0]);
    }

    #[test]
    fn test_combine_clouds_with_offset() {
        let gain_config = GainConfig::default();

        let cloud1 = PointCloud {
            x: vec![1.0],
            y: vec![2.0],
            z: vec![0.0],
            colors: None,
        };

        let cloud2 = PointCloud {
            x: vec![3.0],
            y: vec![4.0],
            z: vec![0.0],
            colors: None,
        };

        let clouds = vec![(75, cloud1), (50, cloud2)];
        let combined = combine_clouds(&clouds, true, &gain_config);

        // Gain 75 has offset 0, gain 50 has offset 250
        assert_eq!(combined.z[0], 0.0); // 0 + 0
        assert_eq!(combined.z[1], 250.0); // 0 + 250
    }

    #[test]
    fn test_build_stacked_clouds() {
        let temp_dir = TempDir::new().unwrap();
        let sweep_dir = temp_dir.path().join("sweeps");
        let output_dir = temp_dir.path().join("output");

        fs::create_dir_all(&sweep_dir).unwrap();

        create_gain_csv(&sweep_dir, 40, 10);
        create_gain_csv(&sweep_dir, 50, 10);

        let config = ProcessingConfig::default();
        let gain_config = GainConfig::default();
        let radar_config = RadarConfig::default();

        let result = build_stacked_clouds(
            &sweep_dir,
            &output_dir,
            &config,
            &gain_config,
            &radar_config,
            true,
            true,
            "test_stack",
        );

        assert!(result.is_ok());
        let outputs = result.unwrap();

        assert!(outputs.contains_key("flat"));
        assert!(outputs.contains_key("offset"));
        assert!(outputs["flat"].exists());
        assert!(outputs["offset"].exists());
    }

    #[test]
    fn test_build_stacked_clouds_empty_dir() {
        let temp_dir = TempDir::new().unwrap();

        let config = ProcessingConfig::default();
        let gain_config = GainConfig::default();
        let radar_config = RadarConfig::default();

        let result = build_stacked_clouds(
            temp_dir.path(),
            &temp_dir.path().join("output"),
            &config,
            &gain_config,
            &radar_config,
            true,
            true,
            "test",
        );

        assert!(result.is_err());
    }
}

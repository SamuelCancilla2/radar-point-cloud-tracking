//! Data loaders for radar CSV and PLY point cloud files.
//!
//! This module provides efficient parsers for:
//! - Radar sweep CSV files (polar format with echo intensity data)
//! - Cartesian point cloud CSV files (x, y, z columns)
//! - ASCII PLY point cloud files (with optional RGB colors)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use csv::ReaderBuilder;
use thiserror::Error;

use crate::config::RadarConfig;

/// Errors that can occur during file loading.
#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV parsing error: {0}")]
    Csv(#[from] csv::Error),

    #[error("Empty file: {0}")]
    EmptyFile(PathBuf),

    #[error("Invalid PLY file: {0}")]
    InvalidPly(String),

    #[error("Missing required columns: {0}")]
    MissingColumns(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Result type for loader operations.
pub type Result<T> = std::result::Result<T, LoaderError>;

/// CSV format detection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsvFormat {
    /// Radar sweep format with Status, Scale, Range, Gain, Angle, Echo_0..Echo_N
    Radar,
    /// Cartesian point cloud format with x, y, z columns
    Cartesian,
}

/// Container for radar sweep data in polar coordinates.
#[derive(Debug, Clone)]
pub struct RadarSweep {
    /// Angle values in radians for each sweep line.
    pub angles_rad: Vec<f32>,
    /// Range values for each (angle, range_bin) pair. Shape: [num_angles][num_bins].
    pub ranges: Vec<Vec<f32>>,
    /// Intensity/echo values for each (angle, range_bin) pair. Shape: [num_angles][num_bins].
    pub intensities: Vec<Vec<f32>>,
    /// Maximum range scale for each angle.
    pub scale: Vec<f32>,
    /// Gain setting if uniform across the sweep.
    pub gain: Option<i32>,
    /// Source file path.
    pub source_path: Option<PathBuf>,
}

impl RadarSweep {
    /// Returns the number of angles (sweep lines) in this sweep.
    #[inline]
    pub fn num_angles(&self) -> usize {
        self.angles_rad.len()
    }

    /// Returns the number of range bins per angle.
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.intensities.first().map_or(0, |row| row.len())
    }
}

/// Container for 3D point cloud data.
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// X coordinates of all points.
    pub x: Vec<f32>,
    /// Y coordinates of all points.
    pub y: Vec<f32>,
    /// Z coordinates of all points.
    pub z: Vec<f32>,
    /// Optional RGB colors for each point.
    pub colors: Option<Vec<[u8; 3]>>,
}

impl PointCloud {
    /// Creates a new empty point cloud.
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            colors: None,
        }
    }

    /// Alias for new() - creates an empty point cloud.
    #[inline]
    pub fn empty() -> Self {
        Self::new()
    }

    /// Creates a new point cloud from coordinate vectors.
    pub fn from_xyz(x: Vec<f32>, y: Vec<f32>, z: Vec<f32>) -> Self {
        Self {
            x,
            y,
            z,
            colors: None,
        }
    }

    /// Creates a new point cloud from coordinate vectors with colors.
    pub fn from_xyz_colors(x: Vec<f32>, y: Vec<f32>, z: Vec<f32>, colors: Vec<[u8; 3]>) -> Self {
        Self {
            x,
            y,
            z,
            colors: Some(colors),
        }
    }

    /// Creates a new point cloud with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
            colors: None,
        }
    }

    /// Returns the number of points in the cloud.
    #[inline]
    pub fn size(&self) -> usize {
        self.x.len()
    }

    /// Alias for size() - returns the number of points.
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Returns true if the point cloud is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Converts point cloud to a vector of [x, y, z] coordinate arrays.
    pub fn to_coords(&self) -> Vec<[f32; 3]> {
        let n = self.size();
        let mut coords = Vec::with_capacity(n);
        for i in 0..n {
            coords.push([self.x[i], self.y[i], self.z[i]]);
        }
        coords
    }

    /// Adds a point to the cloud.
    #[inline]
    pub fn push(&mut self, x: f32, y: f32, z: f32) {
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
    }

    /// Adds a point with color to the cloud.
    pub fn push_with_color(&mut self, x: f32, y: f32, z: f32, color: [u8; 3]) {
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);

        if self.colors.is_none() {
            self.colors = Some(Vec::with_capacity(self.x.capacity()));
        }
        if let Some(ref mut colors) = self.colors {
            colors.push(color);
        }
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}

/// Load radar sweep data from a CSV file.
///
/// The expected CSV format:
/// - Header row (skipped)
/// - Columns: Status, Scale, Range, Gain, Angle, Echo_0, Echo_1, ..., Echo_N
///
/// # Arguments
///
/// * `path` - Path to the radar CSV file
/// * `config` - Radar configuration (uses defaults if None)
///
/// # Returns
///
/// A `RadarSweep` containing angles, ranges, intensities, and metadata.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn load_radar_csv<P: AsRef<Path>>(path: P, config: Option<&RadarConfig>) -> Result<RadarSweep> {
    let path = path.as_ref();
    let default_config = RadarConfig::default();
    let config = config.unwrap_or(&default_config);

    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(BufReader::new(file));

    // Pre-allocate vectors (estimate ~4000 angles per sweep)
    let estimated_rows = 4096;
    let num_echo_cols = config.num_echo_columns;

    let mut angles_rad = Vec::with_capacity(estimated_rows);
    let mut ranges = Vec::with_capacity(estimated_rows);
    let mut intensities = Vec::with_capacity(estimated_rows);
    let mut scale_values = Vec::with_capacity(estimated_rows);
    let mut gain_values = Vec::with_capacity(estimated_rows);

    // Process each record
    for result in reader.records() {
        let record = result?;

        // Parse fixed columns: Status(0), Scale(1), Range(2), Gain(3), Angle(4)
        if record.len() < 5 {
            continue;
        }

        let scale: f32 = record
            .get(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        let gain: i32 = record
            .get(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let angle_raw: f32 = record
            .get(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        // Convert angle to radians: degrees * (360 / 8196) -> radians
        let angle_deg = angle_raw * config.angle_scale;
        let angle_rad = angle_deg.to_radians();

        // Parse echo/intensity values (columns 5 onwards)
        let mut echo_row = Vec::with_capacity(num_echo_cols);
        for i in 0..num_echo_cols {
            let val: f32 = record
                .get(5 + i)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            echo_row.push(val);
        }

        // Compute range values: (scale / num_bins) * bin_index
        let num_bins = echo_row.len();
        let range_step = if num_bins > 0 { scale / num_bins as f32 } else { 0.0 };
        let mut range_row = Vec::with_capacity(num_bins);
        for bin_idx in 0..num_bins {
            range_row.push(range_step * bin_idx as f32);
        }

        angles_rad.push(angle_rad);
        ranges.push(range_row);
        intensities.push(echo_row);
        scale_values.push(scale);
        gain_values.push(gain);
    }

    if angles_rad.is_empty() {
        return Err(LoaderError::EmptyFile(path.to_path_buf()));
    }

    // Determine uniform gain if all values are the same
    let gain = if !gain_values.is_empty() {
        let first = gain_values[0];
        if gain_values.iter().all(|&g| g == first) {
            Some(first)
        } else {
            None
        }
    } else {
        None
    };

    Ok(RadarSweep {
        angles_rad,
        ranges,
        intensities,
        scale: scale_values,
        gain,
        source_path: Some(path.to_path_buf()),
    })
}

/// Load a Cartesian point cloud from a CSV file with x, y, z columns.
///
/// The CSV should have a header row with column names. The function will
/// look for columns named 'x', 'y', 'z' (case-insensitive), or fall back
/// to using the first three columns.
///
/// # Arguments
///
/// * `path` - Path to the CSV file
///
/// # Returns
///
/// A `PointCloud` containing the x, y, z coordinates.
///
/// # Errors
///
/// Returns an error if the file cannot be read or lacks required columns.
pub fn load_cartesian_csv<P: AsRef<Path>>(path: P) -> Result<PointCloud> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(BufReader::new(file));

    // Get headers and map to lowercase
    let headers = reader.headers()?.clone();
    let col_map: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, name)| (name.to_lowercase(), i))
        .collect();

    // Find x, y, z column indices
    let x_idx = col_map.get("x").copied().unwrap_or(0);
    let y_idx = col_map.get("y").copied().unwrap_or(1);
    let z_idx = col_map.get("z").copied().unwrap_or(2);

    // Pre-allocate (estimate ~10000 points)
    let mut cloud = PointCloud::with_capacity(10000);

    for result in reader.records() {
        let record = result?;

        let x: f32 = record
            .get(x_idx)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let y: f32 = record
            .get(y_idx)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let z: f32 = record
            .get(z_idx)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        cloud.push(x, y, z);
    }

    if cloud.is_empty() {
        return Err(LoaderError::EmptyFile(path.to_path_buf()));
    }

    Ok(cloud)
}

/// Load a point cloud from an ASCII PLY file.
///
/// Supports PLY files with vertex elements containing:
/// - Required: x, y, z properties
/// - Optional: red, green, blue color properties
///
/// # Arguments
///
/// * `path` - Path to the PLY file
///
/// # Returns
///
/// A `PointCloud` with coordinates and optional colors.
///
/// # Errors
///
/// Returns an error if the file is not a valid PLY or lacks required properties.
pub fn load_ply<P: AsRef<Path>>(path: P) -> Result<PointCloud> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Check PLY magic number
    let first_line = lines
        .next()
        .ok_or_else(|| LoaderError::InvalidPly("Empty file".to_string()))??;

    if !first_line.trim().starts_with("ply") {
        return Err(LoaderError::InvalidPly(format!(
            "{} is not a PLY file",
            path.display()
        )));
    }

    // Parse header
    let mut num_vertices: Option<usize> = None;
    let mut prop_names: Vec<String> = Vec::new();
    let mut header_done = false;

    for line in &mut lines {
        let line = line?;
        let stripped = line.trim();

        if stripped.starts_with("element vertex") {
            let parts: Vec<&str> = stripped.split_whitespace().collect();
            if let Some(count_str) = parts.last() {
                num_vertices = count_str.parse().ok();
            }
        } else if stripped.starts_with("property") {
            let parts: Vec<&str> = stripped.split_whitespace().collect();
            if let Some(name) = parts.last() {
                prop_names.push(name.to_string());
            }
        } else if stripped == "end_header" {
            header_done = true;
            break;
        }
    }

    let num_vertices = num_vertices
        .ok_or_else(|| LoaderError::InvalidPly("No vertex count in header".to_string()))?;

    if !header_done {
        return Err(LoaderError::InvalidPly(
            "Missing end_header".to_string(),
        ));
    }

    // Build property index map
    let prop_idx: HashMap<&str, usize> = prop_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // Verify required properties
    let x_idx = prop_idx
        .get("x")
        .copied()
        .ok_or_else(|| LoaderError::MissingColumns("x".to_string()))?;
    let y_idx = prop_idx
        .get("y")
        .copied()
        .ok_or_else(|| LoaderError::MissingColumns("y".to_string()))?;
    let z_idx = prop_idx
        .get("z")
        .copied()
        .ok_or_else(|| LoaderError::MissingColumns("z".to_string()))?;

    // Check for color properties
    let has_colors = prop_idx.contains_key("red")
        && prop_idx.contains_key("green")
        && prop_idx.contains_key("blue");

    let (r_idx, g_idx, b_idx) = if has_colors {
        (
            prop_idx["red"],
            prop_idx["green"],
            prop_idx["blue"],
        )
    } else {
        (0, 0, 0)
    };

    // Pre-allocate vectors
    let mut x_vec = Vec::with_capacity(num_vertices);
    let mut y_vec = Vec::with_capacity(num_vertices);
    let mut z_vec = Vec::with_capacity(num_vertices);
    let mut colors_vec = if has_colors {
        Vec::with_capacity(num_vertices)
    } else {
        Vec::new()
    };

    // Parse vertex data
    let mut vertex_count = 0;
    for line in lines {
        if vertex_count >= num_vertices {
            break;
        }

        let line = line?;
        let values: Vec<&str> = line.split_whitespace().collect();

        if values.len() < prop_names.len() {
            continue;
        }

        // Parse coordinates
        let x: f32 = values[x_idx]
            .parse()
            .map_err(|_| LoaderError::ParseError(format!("Invalid x value: {}", values[x_idx])))?;
        let y: f32 = values[y_idx]
            .parse()
            .map_err(|_| LoaderError::ParseError(format!("Invalid y value: {}", values[y_idx])))?;
        let z: f32 = values[z_idx]
            .parse()
            .map_err(|_| LoaderError::ParseError(format!("Invalid z value: {}", values[z_idx])))?;

        x_vec.push(x);
        y_vec.push(y);
        z_vec.push(z);

        // Parse colors if present
        if has_colors {
            let r: u8 = values[r_idx].parse().unwrap_or(180);
            let g: u8 = values[g_idx].parse().unwrap_or(180);
            let b: u8 = values[b_idx].parse().unwrap_or(180);
            colors_vec.push([r, g, b]);
        }

        vertex_count += 1;
    }

    if vertex_count < num_vertices {
        return Err(LoaderError::InvalidPly(format!(
            "Expected {} vertices, found {}",
            num_vertices, vertex_count
        )));
    }

    // If no colors in file, fill with default gray
    let colors = if has_colors {
        Some(colors_vec)
    } else {
        let default_color = [180u8, 180, 180];
        Some(vec![default_color; num_vertices])
    };

    Ok(PointCloud {
        x: x_vec,
        y: y_vec,
        z: z_vec,
        colors,
    })
}

/// Detect whether a CSV file is in radar sweep format or Cartesian format.
///
/// Detection is based on checking for x, y, z column headers.
///
/// # Arguments
///
/// * `path` - Path to the CSV file
///
/// # Returns
///
/// `CsvFormat::Cartesian` if x, y, z columns are found, otherwise `CsvFormat::Radar`.
pub fn detect_csv_format<P: AsRef<Path>>(path: P) -> Result<CsvFormat> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let headers = reader.headers()?;

    // Check for x, y, z columns (case-insensitive)
    let lower_headers: Vec<String> = headers.iter().map(|h| h.to_lowercase()).collect();
    let has_xyz = lower_headers.contains(&"x".to_string())
        && lower_headers.contains(&"y".to_string())
        && lower_headers.contains(&"z".to_string());

    if has_xyz {
        return Ok(CsvFormat::Cartesian);
    }

    // Also check if it's a 3-column file with named columns (not numeric indices)
    if headers.len() == 3 {
        // Check if first header looks like a column name (not a number)
        if let Some(first) = headers.get(0) {
            if first.parse::<f64>().is_err() {
                return Ok(CsvFormat::Cartesian);
            }
        }
    }

    Ok(CsvFormat::Radar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_point_cloud_operations() {
        let mut cloud = PointCloud::new();
        assert!(cloud.is_empty());
        assert_eq!(cloud.size(), 0);

        cloud.push(1.0, 2.0, 3.0);
        cloud.push(4.0, 5.0, 6.0);

        assert_eq!(cloud.size(), 2);
        assert!(!cloud.is_empty());

        let coords = cloud.to_coords();
        assert_eq!(coords.len(), 2);
        assert_eq!(coords[0], [1.0, 2.0, 3.0]);
        assert_eq!(coords[1], [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_point_cloud_with_colors() {
        let mut cloud = PointCloud::new();
        cloud.push_with_color(1.0, 2.0, 3.0, [255, 0, 0]);
        cloud.push_with_color(4.0, 5.0, 6.0, [0, 255, 0]);

        assert_eq!(cloud.size(), 2);
        assert!(cloud.colors.is_some());
        let colors = cloud.colors.unwrap();
        assert_eq!(colors[0], [255, 0, 0]);
        assert_eq!(colors[1], [0, 255, 0]);
    }

    #[test]
    fn test_load_cartesian_csv() -> Result<()> {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "x,y,z").unwrap();
        writeln!(file, "1.0,2.0,3.0").unwrap();
        writeln!(file, "4.0,5.0,6.0").unwrap();
        file.flush().unwrap();

        let cloud = load_cartesian_csv(file.path())?;
        assert_eq!(cloud.size(), 2);
        assert_eq!(cloud.x[0], 1.0);
        assert_eq!(cloud.y[0], 2.0);
        assert_eq!(cloud.z[0], 3.0);

        Ok(())
    }

    #[test]
    fn test_load_ply() -> Result<()> {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "ply").unwrap();
        writeln!(file, "format ascii 1.0").unwrap();
        writeln!(file, "element vertex 2").unwrap();
        writeln!(file, "property float x").unwrap();
        writeln!(file, "property float y").unwrap();
        writeln!(file, "property float z").unwrap();
        writeln!(file, "property uchar red").unwrap();
        writeln!(file, "property uchar green").unwrap();
        writeln!(file, "property uchar blue").unwrap();
        writeln!(file, "end_header").unwrap();
        writeln!(file, "1.0 2.0 3.0 255 0 0").unwrap();
        writeln!(file, "4.0 5.0 6.0 0 255 0").unwrap();
        file.flush().unwrap();

        let cloud = load_ply(file.path())?;
        assert_eq!(cloud.size(), 2);
        assert_eq!(cloud.x[0], 1.0);
        assert_eq!(cloud.y[1], 5.0);

        let colors = cloud.colors.unwrap();
        assert_eq!(colors[0], [255, 0, 0]);
        assert_eq!(colors[1], [0, 255, 0]);

        Ok(())
    }

    #[test]
    fn test_detect_csv_format_cartesian() -> Result<()> {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "x,y,z").unwrap();
        writeln!(file, "1.0,2.0,3.0").unwrap();
        file.flush().unwrap();

        let format = detect_csv_format(file.path())?;
        assert_eq!(format, CsvFormat::Cartesian);

        Ok(())
    }

    #[test]
    fn test_detect_csv_format_radar() -> Result<()> {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Status,Scale,Range,Gain,Angle,Echo_0,Echo_1").unwrap();
        writeln!(file, "1,100,50,40,1000,10,20").unwrap();
        file.flush().unwrap();

        let format = detect_csv_format(file.path())?;
        assert_eq!(format, CsvFormat::Radar);

        Ok(())
    }

    #[test]
    fn test_radar_sweep_methods() {
        let sweep = RadarSweep {
            angles_rad: vec![0.0, 1.57],
            ranges: vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]],
            intensities: vec![vec![10.0, 20.0, 30.0], vec![15.0, 25.0, 35.0]],
            scale: vec![3.0, 3.0],
            gain: Some(40),
            source_path: None,
        };

        assert_eq!(sweep.num_angles(), 2);
        assert_eq!(sweep.num_bins(), 3);
    }
}

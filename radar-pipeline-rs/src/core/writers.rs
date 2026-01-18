//! Data writers for PLY and CSV formats.
//!
//! This module provides functions for writing point cloud data to various file formats:
//! - PLY (Polygon File Format) with ASCII encoding and RGB colors
//! - CSV with Cartesian coordinates
//! - CSV with labeled coordinates for clustering results

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use thiserror::Error;

use super::loaders::PointCloud;

/// Default color for points when no colors are specified (light gray).
const DEFAULT_COLOR: [u8; 3] = [180, 180, 180];

/// Errors that can occur during write operations.
#[derive(Error, Debug)]
pub enum WriteError {
    /// Failed to create parent directories.
    #[error("failed to create parent directories for '{path}': {source}")]
    CreateDirectory {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Failed to create or open file for writing.
    #[error("failed to create file '{path}': {source}")]
    CreateFile {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Failed to write data to file.
    #[error("failed to write to file '{path}': {source}")]
    WriteFile {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// CSV writing error.
    #[error("CSV write error for '{path}': {source}")]
    CsvError {
        path: String,
        #[source]
        source: csv::Error,
    },

    /// Mismatched array lengths.
    #[error("array length mismatch: coords has {coords_len} elements, labels has {labels_len} elements")]
    LengthMismatch { coords_len: usize, labels_len: usize },
}

/// Result type for write operations.
pub type Result<T> = std::result::Result<T, WriteError>;

/// Creates parent directories for a file path if they don't exist.
fn ensure_parent_dirs(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| WriteError::CreateDirectory {
                path: parent.display().to_string(),
                source: e,
            })?;
        }
    }
    Ok(())
}

/// Creates a buffered writer for the given path.
fn create_buffered_writer(path: &Path) -> Result<BufWriter<File>> {
    let file = File::create(path).map_err(|e| WriteError::CreateFile {
        path: path.display().to_string(),
        source: e,
    })?;
    Ok(BufWriter::new(file))
}

/// Write point cloud to ASCII PLY file with RGB colors.
///
/// Creates an ASCII PLY file with the following format:
/// - Header specifying vertex count and properties (x, y, z, red, green, blue)
/// - One line per vertex with space-separated values
///
/// If the point cloud has no colors, a default light gray (180, 180, 180) is used.
///
/// # Arguments
///
/// * `path` - Output file path (parent directories will be created if needed)
/// * `cloud` - Point cloud data with coordinates and optional colors
///
/// # Errors
///
/// Returns an error if:
/// - Parent directories cannot be created
/// - File cannot be created or written to
///
/// # Example
///
/// ```no_run
/// use radar_pipeline::core::loaders::PointCloud;
/// use radar_pipeline::core::writers::write_ply;
/// use std::path::Path;
///
/// let cloud = PointCloud::default();
/// write_ply(Path::new("output.ply"), &cloud).unwrap();
/// ```
pub fn write_ply(path: &Path, cloud: &PointCloud) -> Result<()> {
    ensure_parent_dirs(path)?;
    let mut writer = create_buffered_writer(path)?;

    let num_points = cloud.len();
    let path_str = path.display().to_string();

    // Write PLY header
    writeln!(writer, "ply").map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;
    writeln!(writer, "format ascii 1.0").map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;
    writeln!(writer, "element vertex {}", num_points).map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;
    writeln!(writer, "property float x").map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;
    writeln!(writer, "property float y").map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;
    writeln!(writer, "property float z").map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;
    writeln!(writer, "property uchar red").map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;
    writeln!(writer, "property uchar green").map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;
    writeln!(writer, "property uchar blue").map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;
    writeln!(writer, "end_header").map_err(|e| WriteError::WriteFile {
        path: path_str.clone(),
        source: e,
    })?;

    // Write vertex data
    for i in 0..num_points {
        let x = cloud.x[i];
        let y = cloud.y[i];
        let z = cloud.z[i];
        let [r, g, b] = cloud
            .colors
            .as_ref()
            .map(|c| c[i])
            .unwrap_or(DEFAULT_COLOR);

        writeln!(writer, "{:.6} {:.6} {:.6} {} {} {}", x, y, z, r, g, b).map_err(|e| {
            WriteError::WriteFile {
                path: path_str.clone(),
                source: e,
            }
        })?;
    }

    writer.flush().map_err(|e| WriteError::WriteFile {
        path: path_str,
        source: e,
    })?;

    Ok(())
}

/// Write point cloud to CSV with x, y, z columns.
///
/// Creates a CSV file with headers "x,y,z" and one row per point.
/// Uses a buffered writer for performance.
///
/// # Arguments
///
/// * `path` - Output file path (parent directories will be created if needed)
/// * `cloud` - Point cloud data
///
/// # Errors
///
/// Returns an error if:
/// - Parent directories cannot be created
/// - File cannot be created or written to
///
/// # Example
///
/// ```no_run
/// use radar_pipeline::core::loaders::PointCloud;
/// use radar_pipeline::core::writers::write_cartesian_csv;
/// use std::path::Path;
///
/// let cloud = PointCloud::default();
/// write_cartesian_csv(Path::new("output.csv"), &cloud).unwrap();
/// ```
pub fn write_cartesian_csv(path: &Path, cloud: &PointCloud) -> Result<()> {
    ensure_parent_dirs(path)?;

    let file = File::create(path).map_err(|e| WriteError::CreateFile {
        path: path.display().to_string(),
        source: e,
    })?;
    let buf_writer = BufWriter::new(file);
    let mut csv_writer = csv::Writer::from_writer(buf_writer);

    let path_str = path.display().to_string();

    // Write header
    csv_writer
        .write_record(["x", "y", "z"])
        .map_err(|e| WriteError::CsvError {
            path: path_str.clone(),
            source: e,
        })?;

    // Write data rows
    for i in 0..cloud.len() {
        csv_writer
            .write_record(&[
                format!("{:.6}", cloud.x[i]),
                format!("{:.6}", cloud.y[i]),
                format!("{:.6}", cloud.z[i]),
            ])
            .map_err(|e| WriteError::CsvError {
                path: path_str.clone(),
                source: e,
            })?;
    }

    csv_writer.flush().map_err(|e| WriteError::WriteFile {
        path: path_str,
        source: e,
    })?;

    Ok(())
}

/// Write labeled coordinates to CSV.
///
/// Creates a CSV file with headers "x,y,z,label" containing coordinate data
/// with associated cluster labels. Useful for exporting clustering results.
///
/// # Arguments
///
/// * `path` - Output file path (parent directories will be created if needed)
/// * `coords` - Slice of 3D coordinates as `[x, y, z]` arrays
/// * `labels` - Slice of cluster labels (must have same length as coords)
///
/// # Errors
///
/// Returns an error if:
/// - `coords` and `labels` have different lengths
/// - Parent directories cannot be created
/// - File cannot be created or written to
///
/// # Example
///
/// ```no_run
/// use radar_pipeline::core::writers::write_labels_csv;
/// use std::path::Path;
///
/// let coords = vec![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let labels = vec![0i32, 1];
/// write_labels_csv(Path::new("labels.csv"), &coords, &labels).unwrap();
/// ```
pub fn write_labels_csv(path: &Path, coords: &[[f32; 3]], labels: &[i32]) -> Result<()> {
    // Validate input lengths
    if coords.len() != labels.len() {
        return Err(WriteError::LengthMismatch {
            coords_len: coords.len(),
            labels_len: labels.len(),
        });
    }

    ensure_parent_dirs(path)?;

    let file = File::create(path).map_err(|e| WriteError::CreateFile {
        path: path.display().to_string(),
        source: e,
    })?;
    let buf_writer = BufWriter::new(file);
    let mut csv_writer = csv::Writer::from_writer(buf_writer);

    let path_str = path.display().to_string();

    // Write header
    csv_writer
        .write_record(["x", "y", "z", "label"])
        .map_err(|e| WriteError::CsvError {
            path: path_str.clone(),
            source: e,
        })?;

    // Write data rows
    for (coord, label) in coords.iter().zip(labels.iter()) {
        csv_writer
            .write_record(&[
                format!("{:.6}", coord[0]),
                format!("{:.6}", coord[1]),
                format!("{:.6}", coord[2]),
                label.to_string(),
            ])
            .map_err(|e| WriteError::CsvError {
                path: path_str.clone(),
                source: e,
            })?;
    }

    csv_writer.flush().map_err(|e| WriteError::WriteFile {
        path: path_str,
        source: e,
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_test_cloud() -> PointCloud {
        PointCloud {
            x: vec![1.0, 2.0, 3.0],
            y: vec![4.0, 5.0, 6.0],
            z: vec![7.0, 8.0, 9.0],
            colors: None,
        }
    }

    fn create_test_cloud_with_colors() -> PointCloud {
        PointCloud {
            x: vec![1.0, 2.0],
            y: vec![3.0, 4.0],
            z: vec![5.0, 6.0],
            colors: Some(vec![[255, 0, 0], [0, 255, 0]]),
        }
    }

    #[test]
    fn test_write_ply_without_colors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ply");
        let cloud = create_test_cloud();

        write_ply(&path, &cloud).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines[0], "ply");
        assert_eq!(lines[1], "format ascii 1.0");
        assert_eq!(lines[2], "element vertex 3");
        assert_eq!(lines[9], "end_header");
        // Check first data line uses default color
        assert!(lines[10].contains("180 180 180"));
    }

    #[test]
    fn test_write_ply_with_colors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ply");
        let cloud = create_test_cloud_with_colors();

        write_ply(&path, &cloud).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Check data lines have correct colors
        assert!(lines[10].contains("255 0 0"));
        assert!(lines[11].contains("0 255 0"));
    }

    #[test]
    fn test_write_ply_creates_parent_dirs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("subdir").join("nested").join("test.ply");
        let cloud = create_test_cloud();

        write_ply(&path, &cloud).unwrap();

        assert!(path.exists());
    }

    #[test]
    fn test_write_cartesian_csv() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.csv");
        let cloud = create_test_cloud();

        write_cartesian_csv(&path, &cloud).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines[0], "x,y,z");
        assert_eq!(lines.len(), 4); // header + 3 data rows
    }

    #[test]
    fn test_write_labels_csv() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("labels.csv");
        let coords = vec![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let labels = vec![0i32, 1];

        write_labels_csv(&path, &coords, &labels).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines[0], "x,y,z,label");
        assert_eq!(lines.len(), 3); // header + 2 data rows
        assert!(lines[1].ends_with(",0"));
        assert!(lines[2].ends_with(",1"));
    }

    #[test]
    fn test_write_labels_csv_length_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("labels.csv");
        let coords = vec![[1.0f32, 2.0, 3.0]];
        let labels = vec![0i32, 1]; // Different length

        let result = write_labels_csv(&path, &coords, &labels);

        assert!(result.is_err());
        match result.unwrap_err() {
            WriteError::LengthMismatch {
                coords_len,
                labels_len,
            } => {
                assert_eq!(coords_len, 1);
                assert_eq!(labels_len, 2);
            }
            _ => panic!("Expected LengthMismatch error"),
        }
    }
}

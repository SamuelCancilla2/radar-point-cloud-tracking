//! CSV to Cartesian coordinate conversion.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use thiserror::Error;

use crate::config::RadarConfig;

/// Errors that can occur during Cartesian conversion.
#[derive(Debug, Error)]
pub enum CartesianError {
    #[error("Failed to read input CSV: {0}")]
    ReadError(#[from] std::io::Error),

    #[error("No CSV files found in {folder}")]
    NoFilesFound { folder: PathBuf },

    #[error("Empty CSV file: {path}")]
    EmptyCsv { path: PathBuf },

    #[error("Failed to parse CSV data: {0}")]
    ParseError(String),
}

/// Convert a single radar CSV to Cartesian point CSV.
///
/// Reads a radar sweep CSV, converts polar coordinates to Cartesian,
/// applies an intensity threshold, and writes the result.
///
/// # Arguments
///
/// * `input` - Path to input radar CSV file
/// * `output` - Path to output Cartesian CSV file
/// * `threshold` - Minimum intensity threshold for including points
/// * `config` - Radar configuration
///
/// # Returns
///
/// The number of points written to the output file.
pub fn convert_single_csv(
    input: &Path,
    output: &Path,
    threshold: f32,
    config: &RadarConfig,
) -> Result<usize> {
    // Read input CSV
    let file = File::open(input)
        .with_context(|| format!("Failed to open input file: {}", input.display()))?;
    let reader = BufReader::with_capacity(64 * 1024, file);

    let mut lines = reader.lines();

    // Skip header
    lines.next();

    // Collect data rows
    let rows: Vec<String> = lines.filter_map(|l| l.ok()).collect();

    if rows.is_empty() {
        return Err(CartesianError::EmptyCsv {
            path: input.to_path_buf(),
        }
        .into());
    }

    // Parse rows and convert to Cartesian
    // Pre-allocate vectors for performance
    let estimated_points = rows.len() * config.num_echo_columns;
    let mut x_points = Vec::with_capacity(estimated_points);
    let mut y_points = Vec::with_capacity(estimated_points);
    let mut z_points = Vec::with_capacity(estimated_points);

    let num_angles = rows.len();
    let angle_step = 2.0 * std::f32::consts::PI / num_angles as f32;

    for (angle_idx, row) in rows.iter().enumerate() {
        let fields: Vec<&str> = row.split(',').collect();

        if fields.len() < 6 {
            continue;
        }

        // Parse scale for range calculation
        let scale: f32 = fields[1].trim().parse().unwrap_or(100.0);
        let num_bins = fields.len().saturating_sub(5).min(config.num_echo_columns);

        if num_bins == 0 {
            continue;
        }

        let angle_rad = angle_idx as f32 * angle_step;
        let cos_angle = angle_rad.cos();
        let sin_angle = angle_rad.sin();

        // Process echo columns (starting at index 5)
        for (bin_idx, &field) in fields.iter().skip(5).take(num_bins).enumerate() {
            let intensity: f32 = field.trim().parse().unwrap_or(0.0);

            if intensity > threshold {
                let range = (scale / num_bins as f32) * bin_idx as f32 + config.range_start_m;
                let x = range * cos_angle;
                let y = range * sin_angle;

                x_points.push(x);
                y_points.push(y);
                z_points.push(intensity);
            }
        }
    }

    let num_points = x_points.len();

    // Write output CSV
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    let out_file = File::create(output)
        .with_context(|| format!("Failed to create output file: {}", output.display()))?;
    let mut writer = BufWriter::with_capacity(64 * 1024, out_file);

    writeln!(writer, "x,y,z")?;
    for i in 0..num_points {
        writeln!(writer, "{:.6},{:.6},{:.6}", x_points[i], y_points[i], z_points[i])?;
    }

    writer.flush()?;

    Ok(num_points)
}

/// Iterator over aligned sets of CSV files across gain folders.
///
/// Yields (index, {gain: path}) pairs where files are aligned by their
/// sorted position within each gain folder.
pub struct AlignedInputs {
    lists: HashMap<i32, Vec<PathBuf>>,
    gains: Vec<i32>,
    count: usize,
    current: usize,
}

impl AlignedInputs {
    /// Create a new aligned inputs iterator.
    fn new(base_dir: &Path, gains: &[i32]) -> Result<Self> {
        let mut lists: HashMap<i32, Vec<PathBuf>> = HashMap::with_capacity(gains.len());
        let mut min_count = usize::MAX;

        for &gain in gains {
            let folder = base_dir.join(format!("gain_{}", gain));

            let mut files: Vec<PathBuf> = fs::read_dir(&folder)
                .map_err(|_| CartesianError::NoFilesFound {
                    folder: folder.clone(),
                })?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.extension()
                        .map(|ext| ext.eq_ignore_ascii_case("csv"))
                        .unwrap_or(false)
                })
                .collect();

            if files.is_empty() {
                return Err(CartesianError::NoFilesFound { folder }.into());
            }

            files.sort();
            min_count = min_count.min(files.len());
            lists.insert(gain, files);
        }

        Ok(Self {
            lists,
            gains: gains.to_vec(),
            count: min_count,
            current: 0,
        })
    }
}

impl Iterator for AlignedInputs {
    type Item = (usize, HashMap<i32, PathBuf>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.count {
            return None;
        }

        let idx = self.current;
        self.current += 1;

        let group: HashMap<i32, PathBuf> = self
            .gains
            .iter()
            .filter_map(|&g| {
                self.lists
                    .get(&g)
                    .and_then(|files| files.get(idx).cloned())
                    .map(|path| (g, path))
            })
            .collect();

        Some((idx + 1, group))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count.saturating_sub(self.current);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for AlignedInputs {}

/// Create an iterator over aligned CSV file sets across gain folders.
///
/// # Arguments
///
/// * `base_dir` - Directory containing gain_* subdirectories
/// * `gains` - Gain values to align
///
/// # Returns
///
/// An iterator yielding (index, {gain: path}) pairs.
pub fn aligned_inputs(
    base_dir: &Path,
    gains: &[i32],
) -> Result<impl Iterator<Item = (usize, HashMap<i32, PathBuf>)>> {
    AlignedInputs::new(base_dir, gains)
}

/// Work item for parallel batch conversion.
#[derive(Clone)]
struct ConversionTask {
    idx: usize,
    gain: i32,
    src: PathBuf,
    dest: PathBuf,
}

/// Batch convert aligned gain sweeps to Cartesian CSVs.
///
/// Processes aligned sets of files across gain folders, converting each
/// to Cartesian coordinates in parallel using rayon.
///
/// # Arguments
///
/// * `base_dir` - Directory containing gain_* subdirectories
/// * `output_dir` - Output directory for Cartesian CSVs
/// * `gains` - Gain values to process
/// * `threshold` - Minimum intensity threshold
/// * `limit` - Maximum number of aligned sets to process (None for all)
/// * `config` - Radar configuration
pub fn convert_batch_aligned(
    base_dir: &Path,
    output_dir: &Path,
    gains: &[i32],
    threshold: f32,
    limit: Option<usize>,
    config: &RadarConfig,
) -> Result<()> {
    let inputs = aligned_inputs(base_dir, gains)?;

    // Collect all tasks
    let tasks: Vec<ConversionTask> = inputs
        .take(limit.unwrap_or(usize::MAX))
        .flat_map(|(idx, group)| {
            group.into_iter().map(move |(gain, src)| {
                let out_name = format!("{:04}_gain_{}_cartesian.csv", idx, gain);
                let dest = output_dir.join(format!("gain_{}", gain)).join(out_name);
                ConversionTask {
                    idx,
                    gain,
                    src,
                    dest,
                }
            })
        })
        .collect();

    // Process tasks in parallel
    tasks.par_iter().for_each(|task| {
        match convert_single_csv(&task.src, &task.dest, threshold, config) {
            Ok(n_points) => {
                println!(
                    "[{:04}] gain {}: {} -> {} ({} points)",
                    task.idx,
                    task.gain,
                    task.src.file_name().unwrap_or_default().to_string_lossy(),
                    task.dest.display(),
                    n_points
                );
            }
            Err(e) => {
                eprintln!(
                    "[{:04}] gain {}: Failed to convert {}: {}",
                    task.idx,
                    task.gain,
                    task.src.display(),
                    e
                );
            }
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;
    use tempfile::TempDir;

    fn create_radar_csv(dir: &Path, name: &str, num_rows: usize) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();

        writeln!(file, "Status,Scale,Range,Gain,Angle,Echo_0,Echo_1,Echo_2").unwrap();
        for i in 0..num_rows {
            writeln!(
                file,
                "0,100,50,40,{},{},{},{}",
                i * 45 % 360,
                50 + i,
                60 + i,
                70 + i
            )
            .unwrap();
        }
        path
    }

    fn setup_gain_dirs(base: &Path, gains: &[i32]) {
        for gain in gains {
            fs::create_dir_all(base.join(format!("gain_{}", gain))).unwrap();
        }
    }

    #[test]
    fn test_convert_single_csv() {
        let temp_dir = TempDir::new().unwrap();
        let input = create_radar_csv(temp_dir.path(), "input.csv", 10);
        let output = temp_dir.path().join("output.csv");

        let config = RadarConfig::default();
        let result = convert_single_csv(&input, &output, 0.0, &config);

        assert!(result.is_ok());
        let num_points = result.unwrap();
        assert!(num_points > 0);
        assert!(output.exists());

        // Verify output format
        let content = fs::read_to_string(&output).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines[0].contains("x,y,z"));
        assert!(lines.len() > 1);
    }

    #[test]
    fn test_convert_single_csv_with_threshold() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("input.csv");
        {
            let mut file = File::create(&path).unwrap();
            writeln!(file, "Status,Scale,Range,Gain,Angle,Echo_0,Echo_1").unwrap();
            writeln!(file, "0,100,50,40,0,10,100").unwrap();
        }

        let output = temp_dir.path().join("output.csv");
        let config = RadarConfig::default();

        // With high threshold, should filter out low intensity points
        let result = convert_single_csv(&path, &output, 50.0, &config);
        assert!(result.is_ok());
        let num_points = result.unwrap();
        assert_eq!(num_points, 1); // Only the point with intensity 100
    }

    #[test]
    fn test_aligned_inputs() {
        let temp_dir = TempDir::new().unwrap();
        let gains = vec![40, 50];
        setup_gain_dirs(temp_dir.path(), &gains);

        // Create files in each gain folder
        create_radar_csv(&temp_dir.path().join("gain_40"), "file1.csv", 5);
        create_radar_csv(&temp_dir.path().join("gain_40"), "file2.csv", 5);
        create_radar_csv(&temp_dir.path().join("gain_50"), "file1.csv", 5);
        create_radar_csv(&temp_dir.path().join("gain_50"), "file2.csv", 5);
        create_radar_csv(&temp_dir.path().join("gain_50"), "file3.csv", 5);

        let inputs = aligned_inputs(temp_dir.path(), &gains).unwrap();
        let items: Vec<_> = inputs.collect();

        // Should align to minimum count (2 files in gain_40)
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].0, 1);
        assert_eq!(items[1].0, 2);
        assert_eq!(items[0].1.len(), 2); // Two gains
    }

    #[test]
    fn test_aligned_inputs_missing_folder() {
        let temp_dir = TempDir::new().unwrap();
        let gains = vec![40, 50];

        // Only create gain_40
        fs::create_dir_all(temp_dir.path().join("gain_40")).unwrap();
        create_radar_csv(&temp_dir.path().join("gain_40"), "file1.csv", 5);

        let result = aligned_inputs(temp_dir.path(), &gains);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_batch_aligned() {
        let temp_dir = TempDir::new().unwrap();
        let gains = vec![40, 50];
        setup_gain_dirs(temp_dir.path(), &gains);

        create_radar_csv(&temp_dir.path().join("gain_40"), "file1.csv", 5);
        create_radar_csv(&temp_dir.path().join("gain_50"), "file1.csv", 5);

        let output_dir = temp_dir.path().join("output");
        let config = RadarConfig::default();

        let result =
            convert_batch_aligned(temp_dir.path(), &output_dir, &gains, 0.0, Some(1), &config);

        assert!(result.is_ok());
        assert!(output_dir.join("gain_40").join("0001_gain_40_cartesian.csv").exists());
        assert!(output_dir.join("gain_50").join("0001_gain_50_cartesian.csv").exists());
    }
}

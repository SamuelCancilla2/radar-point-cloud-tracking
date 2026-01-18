//! File sorting by radar gain value.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::Result;
use thiserror::Error;

/// Errors that can occur during file sorting operations.
#[derive(Debug, Error)]
pub enum SortingError {
    #[error("Failed to read CSV file: {0}")]
    CsvReadError(#[from] std::io::Error),

    #[error("Failed to parse gain value from CSV: {path}")]
    GainParseError { path: PathBuf },

    #[error("Directory not found: {0}")]
    DirectoryNotFound(PathBuf),
}

/// Read the first data row and return the Gain column value.
///
/// The Gain value is expected in the 4th column (index 3).
///
/// # Arguments
///
/// * `csv_path` - Path to the CSV file
///
/// # Returns
///
/// The gain value as an integer, or `None` if the file is empty,
/// malformed, or the gain value cannot be parsed.
pub fn sniff_gain(csv_path: &Path) -> Option<i32> {
    let file = File::open(csv_path).ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    lines.next()?;

    // Read first data row
    let row = lines.next()?.ok()?;
    let fields: Vec<&str> = row.split(',').collect();

    // Need at least 4 columns to have Gain (index 3)
    if fields.len() < 4 {
        return None;
    }

    // Parse gain value (4th column, index 3)
    fields[3]
        .trim()
        .parse::<f64>()
        .ok()
        .map(|v| v as i32)
}

/// Sort CSV files by gain value without moving them.
///
/// Scans all CSV files in `source_dir` and groups them by their gain value.
///
/// # Arguments
///
/// * `source_dir` - Directory containing CSV files
/// * `gains` - Gain values to look for
///
/// # Returns
///
/// A HashMap mapping each gain value to a vector of file paths with that gain.
pub fn sort_files_by_gain(
    source_dir: &Path,
    gains: &[i32],
) -> HashMap<i32, Vec<PathBuf>> {
    // Pre-allocate result map with empty vectors for each gain
    let mut result: HashMap<i32, Vec<PathBuf>> = gains
        .iter()
        .map(|&g| (g, Vec::with_capacity(64)))
        .collect();

    // Collect and sort CSV files
    let mut csv_files: Vec<PathBuf> = fs::read_dir(source_dir)
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

    // Categorize files by gain
    for csv_path in csv_files {
        if let Some(gain) = sniff_gain(&csv_path) {
            if let Some(files) = result.get_mut(&gain) {
                files.push(csv_path);
            }
        }
    }

    result
}

/// Sort CSV files into gain-based subdirectories.
///
/// Creates `gain_{value}` subdirectories and moves files into them based
/// on their gain value.
///
/// # Arguments
///
/// * `source_dir` - Directory containing CSV files
/// * `gains` - Gain values to create folders for
/// * `dry_run` - If true, only report what would be moved without actually moving
///
/// # Returns
///
/// A HashMap mapping each gain value to a vector of destination paths
/// (or source paths if dry_run is true).
pub fn move_files_to_gain_folders(
    source_dir: &Path,
    gains: &[i32],
    dry_run: bool,
) -> HashMap<i32, Vec<PathBuf>> {
    // Build target directory names
    let targets: HashMap<i32, String> = gains
        .iter()
        .map(|&g| (g, format!("gain_{}", g)))
        .collect();

    // Pre-allocate result map
    let mut moved: HashMap<i32, Vec<PathBuf>> = gains
        .iter()
        .map(|&g| (g, Vec::with_capacity(64)))
        .collect();

    // Create target directories if not dry run
    if !dry_run {
        for name in targets.values() {
            let dir_path = source_dir.join(name);
            let _ = fs::create_dir_all(&dir_path);
        }
    }

    // Collect and sort CSV files
    let mut csv_files: Vec<PathBuf> = fs::read_dir(source_dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .map(|ext| ext.eq_ignore_ascii_case("csv"))
                    .unwrap_or(false)
        })
        .collect();

    csv_files.sort();

    // Process each file
    for csv_path in csv_files {
        let gain = match sniff_gain(&csv_path) {
            Some(g) => g,
            None => continue,
        };

        let target_name = match targets.get(&gain) {
            Some(name) => name,
            None => continue,
        };

        let file_name = match csv_path.file_name() {
            Some(name) => name,
            None => continue,
        };

        let dest = source_dir.join(target_name).join(file_name);

        if dry_run {
            println!(
                "Would move gain {}: {} -> {}/",
                gain,
                file_name.to_string_lossy(),
                target_name
            );
            if let Some(files) = moved.get_mut(&gain) {
                files.push(csv_path);
            }
        } else {
            match fs::rename(&csv_path, &dest) {
                Ok(()) => {
                    println!(
                        "Moved gain {}: {} -> {}/",
                        gain,
                        file_name.to_string_lossy(),
                        target_name
                    );
                    if let Some(files) = moved.get_mut(&gain) {
                        files.push(dest);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Failed to move {}: {}",
                        file_name.to_string_lossy(),
                        e
                    );
                }
            }
        }
    }

    moved
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_csv(dir: &Path, name: &str, gain: i32) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        writeln!(file, "Status,Scale,Range,Gain,Angle,Echo_0").unwrap();
        writeln!(file, "0,100,50,{},45,128", gain).unwrap();
        path
    }

    #[test]
    fn test_sniff_gain() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = create_test_csv(temp_dir.path(), "test.csv", 50);

        let gain = sniff_gain(&csv_path);
        assert_eq!(gain, Some(50));
    }

    #[test]
    fn test_sniff_gain_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("empty.csv");
        File::create(&csv_path).unwrap();

        let gain = sniff_gain(&csv_path);
        assert_eq!(gain, None);
    }

    #[test]
    fn test_sniff_gain_header_only() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("header_only.csv");
        let mut file = File::create(&csv_path).unwrap();
        writeln!(file, "Status,Scale,Range,Gain,Angle").unwrap();

        let gain = sniff_gain(&csv_path);
        assert_eq!(gain, None);
    }

    #[test]
    fn test_sort_files_by_gain() {
        let temp_dir = TempDir::new().unwrap();
        create_test_csv(temp_dir.path(), "file1.csv", 40);
        create_test_csv(temp_dir.path(), "file2.csv", 50);
        create_test_csv(temp_dir.path(), "file3.csv", 40);
        create_test_csv(temp_dir.path(), "file4.csv", 75);

        let gains = vec![40, 50, 75];
        let result = sort_files_by_gain(temp_dir.path(), &gains);

        assert_eq!(result.get(&40).map(|v| v.len()), Some(2));
        assert_eq!(result.get(&50).map(|v| v.len()), Some(1));
        assert_eq!(result.get(&75).map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_move_files_to_gain_folders_dry_run() {
        let temp_dir = TempDir::new().unwrap();
        create_test_csv(temp_dir.path(), "file1.csv", 40);
        create_test_csv(temp_dir.path(), "file2.csv", 50);

        let gains = vec![40, 50, 75];
        let result = move_files_to_gain_folders(temp_dir.path(), &gains, true);

        assert_eq!(result.get(&40).map(|v| v.len()), Some(1));
        assert_eq!(result.get(&50).map(|v| v.len()), Some(1));

        // Files should still be in original location
        assert!(temp_dir.path().join("file1.csv").exists());
        assert!(temp_dir.path().join("file2.csv").exists());
    }

    #[test]
    fn test_move_files_to_gain_folders() {
        let temp_dir = TempDir::new().unwrap();
        create_test_csv(temp_dir.path(), "file1.csv", 40);
        create_test_csv(temp_dir.path(), "file2.csv", 50);

        let gains = vec![40, 50, 75];
        let result = move_files_to_gain_folders(temp_dir.path(), &gains, false);

        assert_eq!(result.get(&40).map(|v| v.len()), Some(1));
        assert_eq!(result.get(&50).map(|v| v.len()), Some(1));

        // Files should be moved to gain folders
        assert!(temp_dir.path().join("gain_40/file1.csv").exists());
        assert!(temp_dir.path().join("gain_50/file2.csv").exists());

        // Original files should be gone
        assert!(!temp_dir.path().join("file1.csv").exists());
        assert!(!temp_dir.path().join("file2.csv").exists());
    }
}

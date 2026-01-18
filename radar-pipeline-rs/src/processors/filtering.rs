//! File filtering by radar range value.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::Result;
use thiserror::Error;

/// Errors that can occur during filtering operations.
#[derive(Debug, Error)]
pub enum FilteringError {
    #[error("Failed to read CSV file: {0}")]
    CsvReadError(#[from] std::io::Error),

    #[error("Failed to parse range value from CSV: {path}")]
    RangeParseError { path: PathBuf },

    #[error("Directory not found: {0}")]
    DirectoryNotFound(PathBuf),

    #[error("Failed to delete file: {path}")]
    DeleteError { path: PathBuf },
}

/// Read the Range column from the first data row.
///
/// The Range value is expected in the 3rd column (index 2).
///
/// # Arguments
///
/// * `path` - Path to the CSV file
///
/// # Returns
///
/// The range value as an integer, or `None` if the file is empty,
/// malformed, or the range value cannot be parsed.
pub fn get_csv_range(path: &Path) -> Option<i32> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    lines.next()?;

    // Read first data row
    let row = lines.next()?.ok()?;
    let fields: Vec<&str> = row.split(',').collect();

    // Need at least 3 columns to have Range (index 2)
    if fields.len() < 3 {
        return None;
    }

    // Parse range value (3rd column, index 2)
    fields[2]
        .trim()
        .parse::<f64>()
        .ok()
        .map(|v| v as i32)
}

/// Find CSV files in gain subdirectories.
///
/// Searches for `gain_{value}` subdirectories and yields all CSV files found.
///
/// # Arguments
///
/// * `base_dir` - Base directory containing gain_* subdirectories
/// * `gains` - Gain values to search in
///
/// # Returns
///
/// A vector of paths to CSV files found in the gain subdirectories.
pub fn find_targets(base_dir: &Path, gains: &[i32]) -> Vec<PathBuf> {
    // Estimate capacity: assume ~100 files per gain folder
    let mut results = Vec::with_capacity(gains.len() * 100);

    for &gain in gains {
        let folder = base_dir.join(format!("gain_{}", gain));

        if !folder.is_dir() {
            continue;
        }

        let mut csv_files: Vec<PathBuf> = fs::read_dir(&folder)
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
        results.extend(csv_files);
    }

    results
}

/// Find CSV files with specific Range values.
///
/// Searches gain subdirectories and returns paths to CSV files
/// whose Range value matches one of the specified values.
///
/// # Arguments
///
/// * `base_dir` - Base directory containing gain subdirectories
/// * `ranges_to_find` - Set of range values to match
/// * `gains` - Gain values to search in
///
/// # Returns
///
/// A vector of paths to matching CSV files.
pub fn find_files_by_range(
    base_dir: &Path,
    ranges_to_find: &HashSet<i32>,
    gains: &[i32],
) -> Vec<PathBuf> {
    let targets = find_targets(base_dir, gains);
    let mut matches = Vec::with_capacity(targets.len() / 4);

    for path in targets {
        if let Some(range) = get_csv_range(&path) {
            if ranges_to_find.contains(&range) {
                matches.push(path);
            }
        }
    }

    matches
}

/// Delete CSV files with specific Range values.
///
/// Searches gain subdirectories and deletes CSV files whose Range value
/// matches one of the specified values.
///
/// # Arguments
///
/// * `base_dir` - Base directory containing gain subdirectories
/// * `ranges_to_remove` - Set of range values to delete
/// * `gains` - Gain values to search in
/// * `dry_run` - If true, only report what would be deleted without actually deleting
///
/// # Returns
///
/// A vector of paths to deleted (or would-be-deleted) files.
pub fn remove_files_by_range(
    base_dir: &Path,
    ranges_to_remove: &HashSet<i32>,
    gains: &[i32],
    dry_run: bool,
) -> Vec<PathBuf> {
    let to_delete = find_files_by_range(base_dir, ranges_to_remove, gains);

    if to_delete.is_empty() {
        println!("No files with Range in {:?} found.", ranges_to_remove);
        return Vec::new();
    }

    let action = if dry_run { "Would delete" } else { "Deleting" };
    println!("{} {} files:", action, to_delete.len());

    for path in &to_delete {
        println!("  - {}", path.display());

        if !dry_run {
            if let Err(e) = fs::remove_file(path) {
                eprintln!("    Failed to delete: {}", e);
            }
        }
    }

    to_delete
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_csv(dir: &Path, name: &str, range: i32, gain: i32) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        writeln!(file, "Status,Scale,Range,Gain,Angle,Echo_0").unwrap();
        writeln!(file, "0,100,{},{},45,128", range, gain).unwrap();
        path
    }

    fn setup_gain_dirs(base: &Path, gains: &[i32]) {
        for gain in gains {
            fs::create_dir_all(base.join(format!("gain_{}", gain))).unwrap();
        }
    }

    #[test]
    fn test_get_csv_range() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = create_test_csv(temp_dir.path(), "test.csv", 100, 50);

        let range = get_csv_range(&csv_path);
        assert_eq!(range, Some(100));
    }

    #[test]
    fn test_get_csv_range_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("empty.csv");
        File::create(&csv_path).unwrap();

        let range = get_csv_range(&csv_path);
        assert_eq!(range, None);
    }

    #[test]
    fn test_find_targets() {
        let temp_dir = TempDir::new().unwrap();
        let gains = vec![40, 50];
        setup_gain_dirs(temp_dir.path(), &gains);

        create_test_csv(&temp_dir.path().join("gain_40"), "file1.csv", 100, 40);
        create_test_csv(&temp_dir.path().join("gain_40"), "file2.csv", 200, 40);
        create_test_csv(&temp_dir.path().join("gain_50"), "file3.csv", 100, 50);

        let targets = find_targets(temp_dir.path(), &gains);
        assert_eq!(targets.len(), 3);
    }

    #[test]
    fn test_find_targets_missing_dir() {
        let temp_dir = TempDir::new().unwrap();
        let gains = vec![40, 50, 75];

        // Only create gain_40
        fs::create_dir_all(temp_dir.path().join("gain_40")).unwrap();
        create_test_csv(&temp_dir.path().join("gain_40"), "file1.csv", 100, 40);

        let targets = find_targets(temp_dir.path(), &gains);
        assert_eq!(targets.len(), 1);
    }

    #[test]
    fn test_find_files_by_range() {
        let temp_dir = TempDir::new().unwrap();
        let gains = vec![40, 50];
        setup_gain_dirs(temp_dir.path(), &gains);

        create_test_csv(&temp_dir.path().join("gain_40"), "file1.csv", 100, 40);
        create_test_csv(&temp_dir.path().join("gain_40"), "file2.csv", 200, 40);
        create_test_csv(&temp_dir.path().join("gain_50"), "file3.csv", 100, 50);

        let ranges_to_find: HashSet<i32> = [100].into_iter().collect();
        let matches = find_files_by_range(temp_dir.path(), &ranges_to_find, &gains);

        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_remove_files_by_range_dry_run() {
        let temp_dir = TempDir::new().unwrap();
        let gains = vec![40];
        setup_gain_dirs(temp_dir.path(), &gains);

        let file1 = create_test_csv(&temp_dir.path().join("gain_40"), "file1.csv", 100, 40);
        let file2 = create_test_csv(&temp_dir.path().join("gain_40"), "file2.csv", 200, 40);

        let ranges_to_remove: HashSet<i32> = [100].into_iter().collect();
        let deleted = remove_files_by_range(temp_dir.path(), &ranges_to_remove, &gains, true);

        assert_eq!(deleted.len(), 1);
        // Files should still exist in dry run
        assert!(file1.exists());
        assert!(file2.exists());
    }

    #[test]
    fn test_remove_files_by_range() {
        let temp_dir = TempDir::new().unwrap();
        let gains = vec![40];
        setup_gain_dirs(temp_dir.path(), &gains);

        let file1 = create_test_csv(&temp_dir.path().join("gain_40"), "file1.csv", 100, 40);
        let file2 = create_test_csv(&temp_dir.path().join("gain_40"), "file2.csv", 200, 40);

        let ranges_to_remove: HashSet<i32> = [100].into_iter().collect();
        let deleted = remove_files_by_range(temp_dir.path(), &ranges_to_remove, &gains, false);

        assert_eq!(deleted.len(), 1);
        // Only file with range 100 should be deleted
        assert!(!file1.exists());
        assert!(file2.exists());
    }

    #[test]
    fn test_remove_files_no_matches() {
        let temp_dir = TempDir::new().unwrap();
        let gains = vec![40];
        setup_gain_dirs(temp_dir.path(), &gains);

        create_test_csv(&temp_dir.path().join("gain_40"), "file1.csv", 100, 40);

        let ranges_to_remove: HashSet<i32> = [999].into_iter().collect();
        let deleted = remove_files_by_range(temp_dir.path(), &ranges_to_remove, &gains, false);

        assert!(deleted.is_empty());
    }
}

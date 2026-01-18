//! Data processing modules.

pub mod cartesian;
pub mod clustering;
pub mod filtering;
pub mod point_cloud;
pub mod sorting;

// Re-export key types for convenience
pub use cartesian::{aligned_inputs, convert_batch_aligned, convert_single_csv, CartesianError};
pub use filtering::{
    find_files_by_range, find_targets, get_csv_range, remove_files_by_range, FilteringError,
};
pub use point_cloud::{
    apply_gain_colors, build_stacked_clouds, combine_clouds, find_gain_sweeps,
    load_points_from_csv, PointCloud, PointCloudError,
};
pub use sorting::{move_files_to_gain_folders, sniff_gain, sort_files_by_gain, SortingError};

//! High-performance radar point cloud processing pipeline.
//!
//! This crate provides tools for:
//! - Loading and parsing radar CSV and PLY point cloud files
//! - Converting polar radar sweeps to Cartesian coordinates
//! - Building stacked point clouds from multiple gain levels
//! - ST-DBSCAN spatio-temporal clustering (parallelized)
//!
//! # Example
//!
//! ```no_run
//! use radar_pipeline::{core::loaders::load_ply, processors::clustering::cluster_point_cloud};
//!
//! let cloud = load_ply("point_cloud.ply").unwrap();
//! let labels = cluster_point_cloud(&cloud, 5.0, 1.0, 10);
//! ```

pub mod cli;
pub mod config;
pub mod core;
pub mod processors;
pub mod visualization;

pub use config::{ClusteringConfig, GainConfig, PipelineConfig, ProcessingConfig, RadarConfig};
pub use core::loaders::{PointCloud, RadarSweep};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

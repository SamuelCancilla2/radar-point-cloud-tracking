//! Command-line interface for the radar pipeline.

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::{ClusteringConfig, GainConfig};
use crate::PipelineConfig;

#[derive(Parser)]
#[command(name = "radar-pipeline")]
#[command(about = "Radar point cloud processing pipeline", version)]
pub struct Cli {
    /// Path to YAML config file
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Increase verbosity
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Sort CSV files into gain_40/50/75 folders
    SortByGain {
        /// Directory containing CSV files to sort
        directory: PathBuf,
        /// Preview changes without moving files
        #[arg(long)]
        dry_run: bool,
    },

    /// Remove CSV files with specified Range values
    FilterRange {
        /// Directory containing CSV files to filter
        directory: PathBuf,
        /// Range values to filter out
        #[arg(short, long, default_values_t = vec![1, 2])]
        ranges: Vec<i32>,
        /// Preview changes without deleting files
        #[arg(long)]
        dry_run: bool,
    },

    /// Convert radar CSV to Cartesian coordinates
    Convert {
        /// Input CSV file or directory
        input_path: PathBuf,
        /// Output CSV file or directory
        output_path: PathBuf,
        /// Intensity threshold for filtering points
        #[arg(short, long, default_value_t = 0.0)]
        threshold: f32,
        /// Process entire directory (batch mode)
        #[arg(long)]
        batch: bool,
        /// Limit number of files to process
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Build stacked PLY point clouds from Cartesian CSVs
    Build {
        /// Directory containing Cartesian CSV files
        input_dir: PathBuf,
        /// Output directory for PLY files
        output_dir: PathBuf,
        /// Build flat (2D) point cloud
        #[arg(long, default_value_t = true)]
        flat: bool,
        /// Apply Z offset between gain layers
        #[arg(long, default_value_t = true)]
        offset: bool,
    },

    /// Run ST-DBSCAN clustering on point cloud
    Cluster {
        /// Input PLY file
        ply_file: PathBuf,
        /// Output directory for cluster results
        #[arg(short, long)]
        output_dir: Option<PathBuf>,
        /// Spatial epsilon for clustering
        #[arg(long)]
        eps_space: Option<f32>,
        /// Temporal epsilon for clustering
        #[arg(long)]
        eps_time: Option<f32>,
        /// Minimum samples per cluster
        #[arg(long)]
        min_samples: Option<usize>,
        /// Maximum points to process
        #[arg(long)]
        max_points: Option<usize>,
    },

    /// Visualize a point cloud as a 2D scatter plot (PNG)
    Visualize {
        /// Input PLY file
        ply_file: PathBuf,
        /// Output PNG file path (defaults to same name as PLY with .png extension)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Maximum number of points to plot (subsamples if exceeded)
        #[arg(long, default_value_t = 1_000_000)]
        max_points: usize,
        /// Alpha/transparency value for points (0.0 to 1.0)
        #[arg(long, default_value_t = 0.5)]
        alpha: f32,
        /// Title for the plot
        #[arg(long)]
        title: Option<String>,
    },
}

/// Create a spinner for indeterminate operations
fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

/// Print a summary box
fn print_summary(title: &str, items: &[(&str, String)]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ {:<62} ║", title);
    println!("╠══════════════════════════════════════════════════════════════╣");
    for (key, value) in items {
        let display_value = if value.len() > 39 {
            format!("{}...", &value[..36])
        } else {
            value.clone()
        };
        println!("║ {:<20}: {:<39} ║", key, display_value);
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

pub fn run() {
    let cli = Cli::parse();

    // Initialize logging based on verbosity (must come first)
    env_logger::Builder::new()
        .filter_level(match cli.verbose {
            0 => log::LevelFilter::Warn,
            1 => log::LevelFilter::Info,
            _ => log::LevelFilter::Debug,
        })
        .format_timestamp_secs()
        .init();

    // Load config
    let config = match &cli.config {
        Some(path) => {
            match PipelineConfig::from_yaml(path) {
                Ok(cfg) => {
                    info!("Loaded config from: {}", path.display());
                    cfg
                }
                Err(e) => {
                    warn!("Failed to load config from {}: {}, using defaults", path.display(), e);
                    PipelineConfig::default()
                }
            }
        }
        None => PipelineConfig::default(),
    };

    // Dispatch to subcommands
    match cli.command {
        Commands::SortByGain { directory, dry_run } => {
            cmd_sort_by_gain(&directory, dry_run, &config);
        }
        Commands::FilterRange { directory, ranges, dry_run } => {
            cmd_filter_range(&directory, &ranges, dry_run, &config);
        }
        Commands::Convert { input_path, output_path, threshold, batch, limit } => {
            cmd_convert(&input_path, &output_path, threshold, batch, limit, &config);
        }
        Commands::Build { input_dir, output_dir, flat, offset } => {
            cmd_build(&input_dir, &output_dir, flat, offset, &config);
        }
        Commands::Cluster { ply_file, output_dir, eps_space, eps_time, min_samples, max_points } => {
            cmd_cluster(&ply_file, output_dir, eps_space, eps_time, min_samples, max_points, &config);
        }
        Commands::Visualize { ply_file, output, max_points, alpha, title } => {
            cmd_visualize(&ply_file, output, max_points, alpha, title);
        }
    }
}

fn cmd_sort_by_gain(directory: &PathBuf, dry_run: bool, config: &PipelineConfig) {
    use crate::processors::sorting;

    let start = Instant::now();

    if dry_run {
        println!("DRY RUN: No files will be moved");
    }

    let spinner = create_spinner("Scanning directory for CSV files...");

    let gains = &config.gains.values;
    let result = sorting::move_files_to_gain_folders(directory, gains, dry_run);

    spinner.finish_and_clear();

    // Count totals
    let total_files: usize = result.values().map(|v| v.len()).sum();
    let gain_40_count = result.get(&40).map_or(0, |v| v.len());
    let gain_50_count = result.get(&50).map_or(0, |v| v.len());
    let gain_75_count = result.get(&75).map_or(0, |v| v.len());

    print_summary(
        "Sort by Gain Complete",
        &[
            ("Directory", directory.display().to_string()),
            ("Total files", total_files.to_string()),
            ("Gain 40", gain_40_count.to_string()),
            ("Gain 50", gain_50_count.to_string()),
            ("Gain 75", gain_75_count.to_string()),
            ("Dry run", dry_run.to_string()),
            ("Duration", format!("{:.2?}", start.elapsed())),
        ],
    );
}

fn cmd_filter_range(directory: &PathBuf, ranges: &[i32], dry_run: bool, config: &PipelineConfig) {
    use crate::processors::filtering;

    let start = Instant::now();

    if dry_run {
        println!("DRY RUN: No files will be deleted");
    }

    println!("Filtering files with Range values: {:?}", ranges);

    let spinner = create_spinner("Scanning and filtering CSV files...");

    let ranges_set: HashSet<i32> = ranges.iter().copied().collect();
    let gains = &config.gains.values;

    let removed = filtering::remove_files_by_range(directory, &ranges_set, gains, dry_run);

    spinner.finish_and_clear();

    print_summary(
        "Filter by Range Complete",
        &[
            ("Directory", directory.display().to_string()),
            ("Range values", format!("{:?}", ranges)),
            ("Files removed", removed.len().to_string()),
            ("Dry run", dry_run.to_string()),
            ("Duration", format!("{:.2?}", start.elapsed())),
        ],
    );
}

fn cmd_convert(
    input_path: &PathBuf,
    output_path: &PathBuf,
    threshold: f32,
    batch: bool,
    limit: Option<usize>,
    config: &PipelineConfig,
) {
    use crate::processors::cartesian;

    let start = Instant::now();

    // Use config threshold if CLI threshold is default
    let effective_threshold = if threshold == 0.0 {
        config.processing.intensity_threshold
    } else {
        threshold
    };

    if batch {
        // Batch processing mode
        println!("Converting CSV files in batch mode...");
        println!("Input directory: {}", input_path.display());
        println!("Output directory: {}", output_path.display());
        println!("Intensity threshold: {}", effective_threshold);

        if let Some(lim) = limit {
            println!("Processing limit: {} files", lim);
        }

        let gains = &config.gains.values;

        match cartesian::convert_batch_aligned(
            input_path,
            output_path,
            gains,
            effective_threshold,
            limit,
            &config.radar,
        ) {
            Ok(()) => {
                print_summary(
                    "Batch Conversion Complete",
                    &[
                        ("Input directory", input_path.display().to_string()),
                        ("Output directory", output_path.display().to_string()),
                        ("Threshold", effective_threshold.to_string()),
                        ("Duration", format!("{:.2?}", start.elapsed())),
                    ],
                );
            }
            Err(e) => {
                error!("Batch conversion failed: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Single file mode
        println!("Converting single file...");
        println!("Input: {}", input_path.display());
        println!("Output: {}", output_path.display());

        let spinner = create_spinner("Converting to Cartesian coordinates...");

        match cartesian::convert_single_csv(input_path, output_path, effective_threshold, &config.radar) {
            Ok(points) => {
                spinner.finish_and_clear();

                print_summary(
                    "Conversion Complete",
                    &[
                        ("Input file", input_path.display().to_string()),
                        ("Output file", output_path.display().to_string()),
                        ("Points converted", points.to_string()),
                        ("Threshold", effective_threshold.to_string()),
                        ("Duration", format!("{:.2?}", start.elapsed())),
                    ],
                );
            }
            Err(e) => {
                spinner.finish_and_clear();
                error!("Conversion failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn cmd_build(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    flat: bool,
    offset: bool,
    config: &PipelineConfig,
) {
    use crate::processors::point_cloud;

    let start = Instant::now();

    println!("Building stacked point cloud...");
    println!("Input directory: {}", input_dir.display());
    println!("Output directory: {}", output_dir.display());
    println!("Generate flat: {}", flat);
    println!("Generate offset: {}", offset);

    let spinner = create_spinner("Stacking point cloud frames...");

    match point_cloud::build_stacked_clouds(
        input_dir,
        output_dir,
        &config.processing,
        &config.gains,
        &config.radar,
        flat,
        offset,
        "frame_stack",
    ) {
        Ok(outputs) => {
            spinner.finish_and_clear();

            let output_files: Vec<String> = outputs.values().map(|p| p.display().to_string()).collect();

            print_summary(
                "Point Cloud Build Complete",
                &[
                    ("Input directory", input_dir.display().to_string()),
                    ("Output files", output_files.join(", ")),
                    ("Duration", format!("{:.2?}", start.elapsed())),
                ],
            );
        }
        Err(e) => {
            spinner.finish_and_clear();
            error!("Build failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_cluster(
    ply_file: &PathBuf,
    output_dir: Option<PathBuf>,
    eps_space: Option<f32>,
    eps_time: Option<f32>,
    min_samples: Option<usize>,
    max_points: Option<usize>,
    config: &PipelineConfig,
) {
    use crate::processors::clustering;

    let start = Instant::now();

    // Build clustering config with overrides
    let cluster_config = ClusteringConfig {
        eps_space: eps_space.unwrap_or(config.clustering.eps_space),
        eps_time: eps_time.unwrap_or(config.clustering.eps_time),
        min_samples: min_samples.unwrap_or(config.clustering.min_samples),
        max_points: max_points.unwrap_or(config.clustering.max_points),
    };

    // Default output directory to same as input
    let effective_output_dir = output_dir.unwrap_or_else(|| {
        ply_file.parent().unwrap_or(&PathBuf::from(".")).to_path_buf()
    });

    println!("Running ST-DBSCAN clustering...");
    println!("Input: {}", ply_file.display());
    println!("Output directory: {}", effective_output_dir.display());
    println!("Parameters:");
    println!("  eps_space: {}", cluster_config.eps_space);
    println!("  eps_time: {}", cluster_config.eps_time);
    println!("  min_samples: {}", cluster_config.min_samples);
    println!("  max_points: {}", cluster_config.max_points);

    let spinner = create_spinner("Clustering point cloud...");

    let gain_config = GainConfig::default();

    match clustering::process_ply_clustering(
        ply_file,
        Some(&effective_output_dir),
        &cluster_config,
        &gain_config,
    ) {
        Ok((csv_path, labels)) => {
            spinner.finish_and_clear();

            // Count clusters and noise
            let noise_count = labels.iter().filter(|&&l| l == -1).count();
            let cluster_count = labels.iter().filter(|&&l| l >= 0).map(|&l| l).max().unwrap_or(-1) + 1;

            print_summary(
                "Clustering Complete",
                &[
                    ("Input file", ply_file.display().to_string()),
                    ("Output CSV", csv_path.display().to_string()),
                    ("Points processed", labels.len().to_string()),
                    ("Clusters found", cluster_count.to_string()),
                    ("Noise points", noise_count.to_string()),
                    ("eps_space", cluster_config.eps_space.to_string()),
                    ("eps_time", cluster_config.eps_time.to_string()),
                    ("min_samples", cluster_config.min_samples.to_string()),
                    ("Duration", format!("{:.2?}", start.elapsed())),
                ],
            );
        }
        Err(e) => {
            spinner.finish_and_clear();
            error!("Clustering failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_visualize(
    ply_file: &PathBuf,
    output: Option<PathBuf>,
    max_points: usize,
    alpha: f32,
    title: Option<String>,
) {
    use crate::core::loaders;
    use crate::visualization;

    let start = Instant::now();

    // Determine output path (default to same name as input with .png extension)
    let output_path = output.unwrap_or_else(|| {
        let mut path = ply_file.clone();
        path.set_extension("png");
        path
    });

    // Determine title (default to filename)
    let plot_title = title.unwrap_or_else(|| {
        ply_file
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Point Cloud".to_string())
    });

    println!("Visualizing point cloud...");
    println!("Input: {}", ply_file.display());
    println!("Output: {}", output_path.display());
    println!("Max points: {}", max_points);
    println!("Alpha: {}", alpha);

    let spinner = create_spinner("Loading PLY file...");

    // Load the PLY file
    let cloud = match loaders::load_ply(ply_file) {
        Ok(c) => c,
        Err(e) => {
            spinner.finish_and_clear();
            error!("Failed to load PLY file: {}", e);
            std::process::exit(1);
        }
    };

    spinner.set_message("Generating plot...");

    // Generate the plot
    match visualization::plot_point_cloud(&output_path, &cloud, &plot_title, max_points, alpha) {
        Ok(()) => {
            spinner.finish_and_clear();

            print_summary(
                "Visualization Complete",
                &[
                    ("Input file", ply_file.display().to_string()),
                    ("Output PNG", output_path.display().to_string()),
                    ("Points in cloud", cloud.len().to_string()),
                    ("Max points plotted", max_points.to_string()),
                    ("Alpha", alpha.to_string()),
                    ("Duration", format!("{:.2?}", start.elapsed())),
                ],
            );
        }
        Err(e) => {
            spinner.finish_and_clear();
            error!("Visualization failed: {}", e);
            std::process::exit(1);
        }
    }
}

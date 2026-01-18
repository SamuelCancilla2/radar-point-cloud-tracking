//! High-performance ST-DBSCAN clustering for spatio-temporal point clouds.
//!
//! This module implements a parallelized ST-DBSCAN algorithm using:
//! - `kiddo` KD-tree for O(log n) spatial neighbor queries
//! - `rayon` for parallel neighbor finding and core point identification
//! - Atomic union-find for lock-free cluster merging
//!
//! # Example
//!
//! ```no_run
//! use radar_pipeline::processors::clustering::{st_dbscan, infer_time_from_colors};
//! use std::collections::HashMap;
//!
//! let coords = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 100.0, 100.0]];
//! let times = vec![0.0f32, 0.0, 1.0];
//! let labels = st_dbscan(&coords, &times, 5.0, 1.0, 2);
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;

use crate::config::{ClusteringConfig, GainConfig};
use crate::core::loaders::PointCloud;

/// Atomic Union-Find data structure for lock-free parallel cluster merging.
///
/// Uses path compression with atomic compare-and-swap operations to safely
/// merge clusters from multiple threads without locks.
pub struct AtomicUnionFind {
    parent: Vec<AtomicUsize>,
}

impl AtomicUnionFind {
    /// Create a new union-find structure where each element is its own parent.
    #[inline]
    pub fn new(size: usize) -> Self {
        let parent = (0..size)
            .map(|i| AtomicUsize::new(i))
            .collect();
        Self { parent }
    }

    /// Find the root of the set containing `x` with path compression.
    ///
    /// Uses relaxed atomic operations for reading and compare-and-swap
    /// for updates, which is safe because union-find only needs eventual
    /// consistency - we'll always converge to the correct root.
    #[inline]
    pub fn find(&self, mut x: usize) -> usize {
        loop {
            let p = self.parent[x].load(Ordering::Relaxed);
            if p == x {
                return x;
            }
            // Path compression: try to point x directly to grandparent
            let gp = self.parent[p].load(Ordering::Relaxed);
            if gp != p {
                // Attempt path compression (ok if it fails due to concurrent update)
                let _ = self.parent[x].compare_exchange_weak(
                    p,
                    gp,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
            x = p;
        }
    }

    /// Union the sets containing `x` and `y`.
    ///
    /// Uses lock-free compare-and-swap to merge roots. Returns true if
    /// a merge actually occurred, false if they were already in the same set.
    #[inline]
    pub fn union(&self, x: usize, y: usize) -> bool {
        loop {
            let root_x = self.find(x);
            let root_y = self.find(y);

            if root_x == root_y {
                return false;
            }

            // Always make the smaller root point to the larger root
            // This provides some balance without explicit rank tracking
            let (small, large) = if root_x < root_y {
                (root_x, root_y)
            } else {
                (root_y, root_x)
            };

            // Try to update small's parent to large
            match self.parent[small].compare_exchange_weak(
                small,
                large,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(_) => continue, // Retry if another thread modified it
            }
        }
    }
}

/// Infer time indices from point colors by matching to nearest gain colors.
///
/// Each point's RGB color is compared against the known gain colors, and the
/// index of the closest match (in sorted gain order) is returned as the time value.
///
/// # Arguments
///
/// * `colors` - Slice of RGB color triplets, one per point
/// * `gain_colors` - Mapping from gain values to RGB colors
///
/// # Returns
///
/// Vector of time indices (0.0, 1.0, 2.0, ...) corresponding to sorted gain keys.
///
/// # Performance
///
/// This function is parallelized using rayon for large point clouds.
pub fn infer_time_from_colors(
    colors: &[[u8; 3]],
    gain_colors: &HashMap<i32, [u8; 3]>,
) -> Vec<f32> {
    if colors.is_empty() {
        return Vec::new();
    }

    // Sort gains and create palette
    let mut gains_sorted: Vec<i32> = gain_colors.keys().copied().collect();
    gains_sorted.sort_unstable();

    let palette: Vec<[f32; 3]> = gains_sorted
        .iter()
        .map(|g| {
            let c = gain_colors[g];
            [c[0] as f32, c[1] as f32, c[2] as f32]
        })
        .collect();

    // Parallel color matching
    colors
        .par_iter()
        .map(|color| {
            let cf = [color[0] as f32, color[1] as f32, color[2] as f32];

            let mut min_dist = f32::MAX;
            let mut best_idx = 0usize;

            for (idx, pc) in palette.iter().enumerate() {
                let dr = cf[0] - pc[0];
                let dg = cf[1] - pc[1];
                let db = cf[2] - pc[2];
                let dist_sq = dr * dr + dg * dg + db * db;

                if dist_sq < min_dist {
                    min_dist = dist_sq;
                    best_idx = idx;
                }
            }

            best_idx as f32
        })
        .collect()
}

/// Spatio-Temporal DBSCAN clustering algorithm.
///
/// Points are considered neighbors if they are within `eps_space` spatially
/// AND within `eps_time` temporally. A core point has at least `min_samples`
/// neighbors. Clusters are formed by connecting core points that are neighbors.
///
/// # Algorithm (Parallelized)
///
/// 1. **Build KD-tree**: O(n log n) construction using kiddo
/// 2. **Parallel neighbor finding**: Use rayon to query spatial neighbors
///    within eps_space and filter by temporal distance
/// 3. **Parallel core point identification**: A point is core if it has
///    >= min_samples neighbors
/// 4. **Lock-free cluster formation**: Use atomic union-find to merge
///    clusters in parallel - for each core point, union with all core neighbors
/// 5. **Label assignment**: Assign unique cluster IDs from union-find roots;
///    non-core points with no core neighbors get label -1 (noise)
///
/// # Arguments
///
/// * `coords` - Slice of 3D coordinates [x, y, z] per point
/// * `times` - Slice of time values per point
/// * `eps_space` - Spatial neighborhood radius
/// * `eps_time` - Temporal neighborhood threshold
/// * `min_samples` - Minimum points to form a cluster core
///
/// # Returns
///
/// Vector of cluster labels (-1 for noise points).
///
/// # Performance
///
/// - KD-tree queries: O(log n) average per point
/// - Neighbor finding: O(n log n) with parallelization
/// - Union-find operations: O(alpha(n)) amortized (nearly constant)
/// - Overall: O(n log n) with good parallelization
pub fn st_dbscan(
    coords: &[[f32; 3]],
    times: &[f32],
    eps_space: f32,
    eps_time: f32,
    min_samples: usize,
) -> Vec<i32> {
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }

    // Edge case: if min_samples is 0 or 1, every point is its own cluster
    if min_samples <= 1 {
        return (0..n as i32).collect();
    }

    // Phase 1: Build KD-tree from coordinates
    // kiddo's ImmutableKdTree is optimized for batch queries
    let tree: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(coords);

    let eps_space_sq = eps_space * eps_space;

    // Phase 2: Parallel neighbor finding with spatio-temporal filtering
    // For each point, find all neighbors within eps_space spatially
    // and eps_time temporally
    let neighbors: Vec<Vec<usize>> = coords
        .par_iter()
        .enumerate()
        .map(|(i, coord)| {
            let time_i = times[i];

            // Query KD-tree for spatial neighbors within eps_space
            let spatial_neighbors = tree.within::<SquaredEuclidean>(coord, eps_space_sq);

            // Filter by temporal distance
            spatial_neighbors
                .iter()
                .filter_map(|nn| {
                    let idx = nn.item as usize;
                    let time_diff = (times[idx] - time_i).abs();
                    if time_diff <= eps_time {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    // Phase 3: Parallel core point identification
    // A point is a core point if it has >= min_samples neighbors
    let is_core: Vec<bool> = neighbors
        .par_iter()
        .map(|neigh| neigh.len() >= min_samples)
        .collect();

    // Phase 4: Lock-free cluster formation using atomic union-find
    // Only union core points with their core neighbors
    let uf = AtomicUnionFind::new(n);

    // Parallel union of core point clusters
    (0..n).into_par_iter().for_each(|i| {
        if is_core[i] {
            for &j in &neighbors[i] {
                if is_core[j] {
                    uf.union(i, j);
                }
            }
        }
    });

    // Phase 5: Label assignment
    // - Core points: assigned to their union-find root's cluster
    // - Border points (non-core with core neighbor): join first core neighbor's cluster
    // - Noise points (non-core with no core neighbor): label -1

    // First, map union-find roots to sequential cluster IDs
    let mut root_to_cluster: HashMap<usize, i32> = HashMap::new();
    let mut next_cluster_id: i32 = 0;

    // Collect all unique roots from core points
    for i in 0..n {
        if is_core[i] {
            let root = uf.find(i);
            root_to_cluster.entry(root).or_insert_with(|| {
                let id = next_cluster_id;
                next_cluster_id += 1;
                id
            });
        }
    }

    // Assign labels
    let mut labels = vec![-1i32; n];

    for i in 0..n {
        if is_core[i] {
            // Core point: use its cluster
            let root = uf.find(i);
            labels[i] = root_to_cluster[&root];
        } else {
            // Non-core point: try to find a core neighbor
            for &j in &neighbors[i] {
                if is_core[j] {
                    let root = uf.find(j);
                    labels[i] = root_to_cluster[&root];
                    break;
                }
            }
            // If no core neighbor found, label remains -1 (noise)
        }
    }

    labels
}

/// Cluster a point cloud using ST-DBSCAN with color-based time inference.
///
/// This is a convenience wrapper that:
/// 1. Infers time values from point colors using gain color matching
/// 2. Runs ST-DBSCAN clustering on the coordinates with inferred times
///
/// # Arguments
///
/// * `cloud` - Point cloud with coordinates and colors
/// * `config` - Clustering configuration (eps_space, eps_time, min_samples)
/// * `gain_config` - Gain configuration for color-to-time mapping
///
/// # Returns
///
/// Vector of cluster labels (-1 for noise points).
pub fn cluster_point_cloud(
    cloud: &PointCloud,
    config: &ClusteringConfig,
    gain_config: &GainConfig,
) -> Vec<i32> {
    // Extract coordinates as contiguous array using the PointCloud's to_coords method
    let coords = cloud.to_coords();

    // Infer time from colors
    let times = if let Some(ref colors) = cloud.colors {
        infer_time_from_colors(colors, &gain_config.colors)
    } else {
        // No colors - treat all points as same time
        vec![0.0f32; coords.len()]
    };

    st_dbscan(
        &coords,
        &times,
        config.eps_space,
        config.eps_time,
        config.min_samples,
    )
}

/// Process a PLY file: load, optionally subsample, cluster, and save results.
///
/// # Arguments
///
/// * `ply_path` - Path to input PLY file
/// * `output_dir` - Output directory (defaults to PLY parent directory)
/// * `config` - Clustering configuration
/// * `gain_config` - Gain configuration for color-to-time mapping
///
/// # Returns
///
/// Tuple of (output CSV path, cluster labels vector).
///
/// # Errors
///
/// Returns error if PLY loading or file writing fails.
pub fn process_ply_clustering(
    ply_path: &Path,
    output_dir: Option<&Path>,
    config: &ClusteringConfig,
    gain_config: &GainConfig,
) -> Result<(PathBuf, Vec<i32>), Box<dyn std::error::Error>> {
    use crate::core::loaders::load_ply;
    use crate::core::transforms::subsample_cloud;
    use crate::core::writers::write_labels_csv;

    // Load PLY file
    let cloud = load_ply(ply_path)?;

    // Subsample if needed
    let (cloud, stride) = subsample_cloud(&cloud, config.max_points);
    let file_name = ply_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    log::info!(
        "{}: using {} points (approx stride={})",
        file_name,
        cloud.size(),
        stride
    );

    // Run clustering
    let labels = cluster_point_cloud(&cloud, config, gain_config);

    // Compute summary statistics
    let mut label_counts: HashMap<i32, usize> = HashMap::new();
    for &label in &labels {
        *label_counts.entry(label).or_insert(0) += 1;
    }
    log::info!("{}: cluster summary {:?}", file_name, label_counts);

    // Determine output directory
    let out_dir = output_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| ply_path.parent().unwrap_or(Path::new(".")).to_path_buf());

    // Generate output path
    let stem = ply_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let csv_path = out_dir.join(format!("{}_dbscan_labels.csv", stem));

    // Write results
    let coords = cloud.to_coords();
    write_labels_csv(&csv_path, &coords, &labels)?;
    log::info!("Labels CSV -> {}", csv_path.display());

    Ok((csv_path, labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_union_find_basic() {
        let uf = AtomicUnionFind::new(5);

        // Initially each element is its own root
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(4), 4);

        // Union 0 and 1
        assert!(uf.union(0, 1));
        assert_eq!(uf.find(0), uf.find(1));

        // Union 2 and 3
        assert!(uf.union(2, 3));
        assert_eq!(uf.find(2), uf.find(3));

        // They should be in different sets
        assert_ne!(uf.find(0), uf.find(2));

        // Union the two sets
        assert!(uf.union(1, 2));
        assert_eq!(uf.find(0), uf.find(3));

        // Union of same set returns false
        assert!(!uf.union(0, 3));
    }

    #[test]
    fn test_infer_time_from_colors() {
        let mut gain_colors = HashMap::new();
        gain_colors.insert(40, [0u8, 114, 255]); // blue
        gain_colors.insert(50, [0u8, 200, 83]); // green
        gain_colors.insert(75, [255u8, 87, 34]); // orange

        // Exact matches
        let colors = vec![
            [0u8, 114, 255], // blue -> gain 40 -> index 0
            [0u8, 200, 83],  // green -> gain 50 -> index 1
            [255u8, 87, 34], // orange -> gain 75 -> index 2
        ];

        let times = infer_time_from_colors(&colors, &gain_colors);
        assert_eq!(times.len(), 3);
        assert_eq!(times[0], 0.0);
        assert_eq!(times[1], 1.0);
        assert_eq!(times[2], 2.0);

        // Near matches (should still work)
        let colors_near = vec![
            [5u8, 110, 250], // near blue
            [10u8, 195, 88], // near green
        ];

        let times_near = infer_time_from_colors(&colors_near, &gain_colors);
        assert_eq!(times_near[0], 0.0); // nearest to blue
        assert_eq!(times_near[1], 1.0); // nearest to green
    }

    #[test]
    fn test_st_dbscan_simple_clusters() {
        // Create two clear clusters separated in space
        let coords: Vec<[f32; 3]> = vec![
            // Cluster 1: around origin
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            // Cluster 2: far away
            [100.0, 100.0, 0.0],
            [101.0, 100.0, 0.0],
            [100.0, 101.0, 0.0],
            [101.0, 101.0, 0.0],
        ];
        let times = vec![0.0f32; 8]; // All same time

        let labels = st_dbscan(&coords, &times, 5.0, 1.0, 2);

        assert_eq!(labels.len(), 8);

        // First 4 points should be in same cluster
        assert!(labels[0] >= 0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);

        // Last 4 points should be in same cluster
        assert!(labels[4] >= 0);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);

        // But different from first cluster
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_st_dbscan_temporal_separation() {
        // Points close in space but separated in time
        let coords: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let times = vec![0.0, 0.0, 5.0, 5.0]; // Two time groups

        // With small eps_time, should form two clusters
        let labels = st_dbscan(&coords, &times, 5.0, 1.0, 2);

        assert_eq!(labels[0], labels[1]); // Same time
        assert_eq!(labels[2], labels[3]); // Same time
        assert_ne!(labels[0], labels[2]); // Different times
    }

    #[test]
    fn test_st_dbscan_noise_points() {
        let coords: Vec<[f32; 3]> = vec![
            // Dense cluster
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            // Isolated point (noise)
            [100.0, 100.0, 100.0],
        ];
        let times = vec![0.0f32; 4];

        let labels = st_dbscan(&coords, &times, 5.0, 1.0, 3);

        // First 3 points should be clustered
        assert!(labels[0] >= 0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);

        // Last point should be noise
        assert_eq!(labels[3], -1);
    }

    #[test]
    fn test_st_dbscan_empty() {
        let coords: Vec<[f32; 3]> = vec![];
        let times: Vec<f32> = vec![];

        let labels = st_dbscan(&coords, &times, 5.0, 1.0, 3);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_st_dbscan_single_point() {
        let coords = vec![[0.0f32, 0.0, 0.0]];
        let times = vec![0.0f32];

        let labels = st_dbscan(&coords, &times, 5.0, 1.0, 2);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], -1); // Single point is noise if min_samples > 1
    }

    #[test]
    fn test_infer_time_empty() {
        let colors: Vec<[u8; 3]> = vec![];
        let gain_colors = HashMap::new();

        let times = infer_time_from_colors(&colors, &gain_colors);
        assert!(times.is_empty());
    }
}

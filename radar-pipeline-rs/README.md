# Radar Pipeline (Rust)

High-performance radar point cloud processing pipeline written in Rust with parallelization via Rayon.

## Features

- **Parallel Processing**: Uses Rayon for data parallelism across all CPU cores
- **Lock-free ST-DBSCAN**: Parallel spatio-temporal clustering using atomic union-find
- **KD-tree Spatial Indexing**: O(log n) neighbor queries via kiddo
- **Efficient I/O**: Buffered reading/writing for CSV and PLY files
- **Configurable**: YAML configuration files with sensible defaults

## Installation

```bash
cargo build --release
```

## CLI Commands

```bash
# Sort CSV files by gain value into gain_40/50/75 folders
radar-pipeline sort-by-gain ./data/raw

# Remove CSV files with Range values 1 and 2
radar-pipeline filter-range ./data/sorted --ranges 1 2

# Convert radar CSV to Cartesian coordinates
radar-pipeline convert input.csv output.csv --threshold 10.0

# Batch convert aligned CSVs
radar-pipeline convert ./data/sorted ./output --batch

# Build stacked PLY point clouds
radar-pipeline build ./output/cartesian ./output/ply

# Run ST-DBSCAN clustering
radar-pipeline cluster ./output/ply/stack.ply --eps-space 5.0 --eps-time 1.0
```

## Configuration

Create a `config.yaml` file:

```yaml
gains:
  values: [40, 50, 75]
  colors:
    40: [0, 114, 255]
    50: [0, 200, 83]
    75: [255, 87, 34]

radar:
  angle_scale: 0.043945
  num_echo_columns: 1024
  range_bin_width_m: 0.5

processing:
  intensity_threshold: 0.0
  point_stride: 16
  max_points_per_gain: 10000000

clustering:
  eps_space: 5.0
  eps_time: 1.0
  min_samples: 10
  max_points: 10000000
```

Use with: `radar-pipeline -c config.yaml <command>`

## Performance

The Rust implementation provides significant speedups over the Python version:
- **Parallel polar-to-Cartesian conversion** using Rayon
- **Lock-free ST-DBSCAN clustering** with atomic union-find
- **Zero-copy parsing** where possible
- **Pre-allocated buffers** to minimize allocations

## License

MIT

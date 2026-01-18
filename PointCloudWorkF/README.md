# Radar Point Cloud Denoising with ST-DBSCAN

A system for processing marine radar data into point clouds and applying Spatio-Temporal DBSCAN (ST-DBSCAN) for noise suppression and object clustering.

## Abstract

Radar data can be noisy and unpredictable with various environmental factors contributing to clutter and uncertainty. This makes it difficult to track objects over time. Moving objects blip in and out of frames, noise is mistaken for objects, and meshing between objects with high reflectivity and low reflectivity can occur.

This work presents a solution through **Spatio-Temporal Density-Based Clustering of Applications with Noise (ST-DBSCAN)** applied to raw radar data which is then converted into point cloud form.

The pipeline first transforms Status, Scale, Range, Gain, Angle, and Echo Values from each radar frame into Cartesian coordinates and then point cloud data. When the data is in point cloud form, ST-DBSCAN is applied to cluster consistent object returns while suppressing noise.

This method produces more correct object groupings across sequential frames and reduces false positives caused by noise. The results demonstrate an effective approach for improving object denoising in a radar system.

## Project Structure

```
├── PointCloudWorkF/
│   ├── stdbscan_denoising_pipeline.py   # Main denoising pipeline
│   ├── denoising_results/               # Output folder (generated)
│   └── README.md                         # This file
│
├── (.125NM)data_pattern3(.125NM)/        # Raw radar data
│   ├── gain_40/                          # Radar CSVs at gain 40
│   ├── gain_50/                          # Radar CSVs at gain 50
│   └── gain_75/                          # Radar CSVs at gain 75
│
└── junk/                                 # Old/unused files
```

## Quick Start

### Just Click Run (Recommended)

Simply open `stdbscan_denoising_pipeline.py` in your IDE and click **Run**. No arguments needed!

This will:
- Process **5 frames** of radar data automatically
- Generate cluster visualizations (PNG images)
- Save results to `denoising_results/` folder
- Open the output folder when complete (Windows)

### Command Line Usage

```bash
# Quick run (5 frames, memory-safe) - same as clicking Run
python stdbscan_denoising_pipeline.py

# Process more frames
python stdbscan_denoising_pipeline.py --max-frames 20

# Process all frames (careful with memory!)
python stdbscan_denoising_pipeline.py --max-frames 0

# Custom clustering parameters
python stdbscan_denoising_pipeline.py --eps-space 10.0 --min-samples 20

# Memory-constrained systems (WSL, low RAM)
python stdbscan_denoising_pipeline.py --low-memory --no-parallel

# Skip visualizations for faster processing
python stdbscan_denoising_pipeline.py --max-frames 50 --no-viz
```

## Running Experiments

The `run_experiments.py` script automates parameter comparison by running multiple ST-DBSCAN configurations and generating a comparison report.

### Quick Start

Simply open `run_experiments.py` in your IDE and click **Run**. By default, it runs in quick mode with a single experiment.

### Configuration

Edit the settings at the top of the script:

```python
# Number of frames to process (lower = faster)
MAX_FRAMES = 10

# True = single quick test, False = run all experiments
QUICK_MODE = True
```

### Command Line Usage

```bash
# Run with default settings (quick mode)
python run_experiments.py
```

### Output

The script generates:
- `experiment_results.json` - Raw results data
- `stdbscan_comparison_report.tex` - LaTeX report with comparison tables
- `results_<experiment_name>/` - Output folder for each experiment

### Experiment Configurations

| Name | eps_space | eps_time | min_samples | min_frames | Description |
|------|-----------|----------|-------------|------------|-------------|
| default | 8.0 | 2.0 | 15 | 2 | Default parameters |
| tight_spatial | 5.0 | 2.0 | 15 | 2 | Tighter spatial radius (5m) |
| aggressive | 5.0 | 1.5 | 25 | 3 | Aggressive filtering |

To add custom experiments, modify the `FULL_EXPERIMENTS` list in the script and set `QUICK_MODE = False`.

## Pipeline Stages

### Stage 1: Radar Data Loading
- Reads raw radar CSV files organized by gain level
- CSV format: Status, Scale, Range, Gain, Angle, Echo_0...Echo_1023
- Each row represents one angular sweep with 1024 range bins

### Stage 2: Coordinate Conversion
- Converts polar coordinates (angle, range) to Cartesian (x, y)
- Applies intensity threshold to filter weak returns
- Groups files from different gains into temporal frames

### Stage 3: ST-DBSCAN Clustering
- Applies Spatio-Temporal DBSCAN for noise suppression
- Points are clustered if they are:
  - Within `eps_space` meters of each other (spatial proximity)
  - Within `eps_time` frames of each other (temporal proximity)
- Points not meeting cluster criteria are classified as noise

### Stage 4: Output Generation
- Denoised point cloud (PLY format)
- Raw vs denoised comparison visualization
- Cluster statistics and noise reduction metrics

## Output Files

| File | Description |
|------|-------------|
| `denoised_point_cloud.ply` | Cleaned point cloud with noise removed |
| `raw_point_cloud.ply` | Original point cloud before denoising |
| `denoising_comparison.png` | Side-by-side before/after visualization |
| `temporal_clusters.png` | Cluster visualization across time frames |
| `noise_reduction_stats.png` | Statistics showing noise reduction |
| `denoising_stats.csv` | Numerical denoising statistics |
| `clusters.csv` | Information about detected clusters |

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eps-space` | 8.0 | Spatial clustering radius (meters) |
| `--eps-time` | 2.0 | Temporal clustering window (frames) |
| `--min-samples` | 15 | Minimum points to form a cluster |
| `--max-frames` | 5 | Limit number of frames to process (0 = all) |
| `--no-viz` | false | Skip visualization generation |
| `--low-memory` | false | Memory-efficient mode (frees intermediate data) |
| `--no-parallel` | false | Disable parallel CSV loading (uses less RAM) |

### Parameter Tuning Guide

**eps-space** (spatial radius):
- Increase if objects are sparse or detections are being missed
- Decrease if separate objects are merging into single clusters

**eps-time** (temporal window):
- Increase to connect objects that appear intermittently
- Decrease if different time periods are incorrectly merging

**min-samples** (cluster threshold):
- Increase to filter out more noise (fewer, more confident detections)
- Decrease to detect smaller objects

## Data Format

**Input:** Raw radar CSV files with columns:
- Column 0: Status (radar status code)
- Column 1: Scale (maximum range in meters)
- Column 2: Range (range setting)
- Column 3: Gain (gain level: 40, 50, or 75 dB)
- Column 4: Angle (0-8196 radar units = 0-360 degrees)
- Columns 5-1028: Echo_0 to Echo_1023 (intensity values 0-255)

**Output:** PLY point clouds (binary format for speed) viewable in:
- CloudCompare (recommended)
- MeshLab
- Blender
- Open3D (Python)

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

## License

This project is for research and educational purposes.

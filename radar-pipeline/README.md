# Radar Pipeline

A Python package for processing radar point cloud data, performing coordinate conversions, building stacked point clouds, and running ST-DBSCAN clustering.

## Features

- **File Organization**: Sort radar CSV files by gain value
- **Range Filtering**: Remove files with specific range values
- **Coordinate Conversion**: Convert polar radar data to Cartesian coordinates
- **Point Cloud Building**: Create stacked PLY point clouds from multiple gain levels
- **Visualization**: Generate PNG previews of point clouds
- **Clustering**: ST-DBSCAN spatio-temporal clustering

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bash
# Sort files by gain
radar-pipeline sort-by-gain ./data/raw

# Remove files with Range 1 or 2
radar-pipeline filter-range ./data/sorted --ranges 1 --ranges 2

# Convert single CSV to Cartesian
radar-pipeline convert ./data/raw/sweep.csv ./output/cartesian.csv -t 0.0

# Batch convert aligned gains
radar-pipeline convert ./data/sorted ./output --batch --limit 10

# Build point clouds
radar-pipeline build ./output/cartesian ./output/ply --plot

# Generate PNG preview
radar-pipeline visualize ./output/ply/frame_stack_v3.ply --max-points 500000

# Run clustering
radar-pipeline cluster ./output/ply/frame_stack_v3.ply --eps-space 5.0 --min-samples 10

# Use custom config
radar-pipeline -c config/default.yaml build ./input ./output
```

### Python API

```python
from radar_pipeline.core import load_radar_csv, load_ply
from radar_pipeline.processors import convert_single_csv, st_dbscan
from radar_pipeline.config import PipelineConfig

# Load configuration
config = PipelineConfig()

# Load radar data
sweep = load_radar_csv("path/to/sweep.csv")

# Convert to Cartesian
convert_single_csv("input.csv", "output.csv", threshold=0.0)

# Load point cloud
cloud = load_ply("path/to/cloud.ply")

# Run clustering
labels = st_dbscan(coords, times, eps_space=5.0, eps_time=1.0, min_samples=10)
```

## Configuration

Configuration can be provided via YAML file or programmatically:

```yaml
gains:
  values: [40, 50, 75]
  colors:
    40: [0, 114, 255]
    50: [0, 200, 83]
    75: [255, 87, 34]

processing:
  intensity_threshold: 0.0
  point_stride: 16
  max_points_per_gain: 10000000

clustering:
  eps_space: 5.0
  eps_time: 1.0
  min_samples: 10
```

## Pipeline Stages

1. **Sort by Gain** (`sort-by-gain`): Organize raw CSVs into gain_40/, gain_50/, gain_75/ folders
2. **Filter Range** (`filter-range`): Remove files with unwanted range values
3. **Convert** (`convert`): Transform polar radar data to Cartesian coordinates
4. **Build** (`build`): Create stacked PLY point clouds
5. **Visualize** (`visualize`): Generate PNG previews
6. **Cluster** (`cluster`): Run ST-DBSCAN clustering

## Project Structure

```
radar-pipeline/
├── src/radar_pipeline/
│   ├── config/          # Configuration models
│   ├── core/            # Data loaders, writers, transforms
│   ├── processors/      # Pipeline stage implementations
│   ├── visualization/   # Plotting functions
│   └── cli/             # Command-line interface
├── tests/               # Test suite
└── config/              # Default configuration files
```

## License

MIT License

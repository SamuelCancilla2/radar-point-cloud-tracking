# Radar Point Cloud Processing Pipeline

A comprehensive system for processing marine radar data into point clouds, detecting and tracking objects (buoys and boats) across time.

## Project Structure

```
├── PointCloudWork/                      # Main Python scripts
│   ├── 1_CSVtoCartesian.py             # Convert radar CSV to Cartesian coordinates
│   ├── 2_build_point_clouds.py         # Build PLY point clouds from Cartesian data
│   ├── 2.5_point_cloud_png_generator.py # Generate PNG previews
│   ├── 3_stdbscan_point_clouds.py      # ST-DBSCAN clustering on point clouds
│   ├── 4_temporal_object_tracker.py    # Object tracking system (NEW)
│   └── 5_gain_fusion_ply_builder.py    # Gain-fused PLY builder (NEW)
│
├── radar-pipeline/                      # Refactored Python package
│   └── src/radar_pipeline/             # Modular pipeline implementation
│
├── radar-pipeline-rs/                   # High-performance Rust implementation
│   └── src/                            # Rust source files
│
└── (.125NM)data_pattern3(.125NM)/      # Data directory (CSVs not tracked)
    ├── gain_40/                        # Radar data at gain 40
    ├── gain_50/                        # Radar data at gain 50
    └── gain_75/                        # Radar data at gain 75
```

## New Features (Object Tracking System)

### 4_temporal_object_tracker.py

A complete object detection and tracking pipeline that:

1. **Fuses multiple gains** (40, 50, 70/75) into unified point clouds per time frame
2. **Filters out land/stationary background** using persistence analysis
3. **Applies temporal ST-DBSCAN** for spatiotemporal clustering
4. **Classifies objects** as:
   - **Buoys** (stationary objects)
   - **Boats** (moving objects)
5. **Tracks objects** with persistent IDs using Hungarian algorithm

```bash
# Run with default settings
python PointCloudWork/4_temporal_object_tracker.py

# Process first 100 frames with visualization
python PointCloudWork/4_temporal_object_tracker.py --max-frames 100

# Skip land filtering
python PointCloudWork/4_temporal_object_tracker.py --no-land-filter

# Custom clustering parameters
python PointCloudWork/4_temporal_object_tracker.py --eps-space 10.0 --min-samples 20
```

**Output files:**
- `tracked_objects.csv` - Summary of all tracked objects
- `trajectories.csv` - Position history for each object
- `clusters.csv` - Cluster details per frame
- `visualizations/` - PNG images of frames with labeled objects

### 5_gain_fusion_ply_builder.py

Creates gain-fused PLY point clouds with absolute intensity values:

```bash
# Build individual frame PLYs
python PointCloudWork/5_gain_fusion_ply_builder.py individual --max-frames 50

# Build temporal stack (all frames in one PLY)
python PointCloudWork/5_gain_fusion_ply_builder.py stacked --max-frames 100

# Compare gains at a single frame
python PointCloudWork/5_gain_fusion_ply_builder.py comparison --frame 0
```

## Pipeline Stages

1. **File Organization** - Sort raw CSVs by gain value
2. **Range Filtering** - Remove unwanted range values
3. **Coordinate Conversion** - Convert polar to Cartesian coordinates
4. **Point Cloud Building** - Create PLY files from point data
5. **Clustering** - Apply ST-DBSCAN for object detection
6. **Tracking** - Track objects across frames with persistent IDs

## Data Format

**Input:** Radar CSV files with columns:
- Status, Scale, Range, Gain, Angle, Echo_0...Echo_1023

**Output:**
- PLY point clouds (ASCII format with RGB colors)
- CSV files with tracking results
- PNG visualizations

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

## Configuration

Key parameters can be adjusted via command-line arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eps-space` | 8.0 | Spatial clustering radius (meters) |
| `--eps-time` | 2.0 | Temporal clustering window (frames) |
| `--min-samples` | 15 | Minimum points per cluster |
| `--intensity-threshold` | 10.0 | Minimum intensity to keep |

## Object Classification

Objects are classified based on motion analysis:
- **Buoy**: Average velocity < 1.0 m/frame
- **Boat**: Average velocity >= 1.0 m/frame
- **Unknown**: Insufficient tracking history

## License

This project is for research and educational purposes.

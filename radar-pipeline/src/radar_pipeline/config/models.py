"""Pydantic configuration models for radar pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from pydantic import BaseModel, Field


class GainConfig(BaseModel):
    """Configuration for radar gain levels."""

    values: Tuple[int, ...] = (40, 50, 75)
    colors: Dict[int, Tuple[int, int, int]] = Field(
        default={
            40: (0, 114, 255),   # blue
            50: (0, 200, 83),    # green
            75: (255, 87, 34),   # orange
        }
    )
    z_offsets: Dict[int, float] = Field(
        default={
            75: 0.0,    # bottom layer
            50: 250.0,  # middle layer
            40: 500.0,  # top layer
        }
    )


class RadarConfig(BaseModel):
    """Configuration for radar data parameters."""

    angle_scale: float = 360.0 / 8196.0
    num_echo_columns: int = 1024
    range_bin_width_m: float = 0.5
    range_start_m: float = 0.0


class ProcessingConfig(BaseModel):
    """Configuration for point cloud processing."""

    intensity_threshold: float = 0.0
    point_stride: int = 16
    max_points_per_gain: int = 10_000_000
    max_points_stack: int = 20_000_000
    plot_max_points: int = 1_000_000


class ClusteringConfig(BaseModel):
    """Configuration for ST-DBSCAN clustering."""

    eps_space: float = 5.0
    eps_time: float = 1.0
    min_samples: int = 10
    max_points: int = 10_000_000


class PipelineConfig(BaseModel):
    """Main pipeline configuration combining all sub-configs."""

    gains: GainConfig = Field(default_factory=GainConfig)
    radar: RadarConfig = Field(default_factory=RadarConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        import yaml

        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False)

"""CSV to Cartesian coordinate conversion."""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from ..config import GainConfig, RadarConfig
from ..core.loaders import load_radar_csv
from ..core.transforms import polar_to_cartesian


def convert_single_csv(
    input_path: Path,
    output_path: Path,
    threshold: float = 0.0,
    config: Optional[RadarConfig] = None,
) -> int:
    """
    Convert single radar CSV to Cartesian point CSV.

    Parameters
    ----------
    input_path : Path
        Input radar CSV file.
    output_path : Path
        Output Cartesian CSV file.
    threshold : float
        Minimum intensity threshold.
    config : RadarConfig, optional
        Radar configuration.

    Returns
    -------
    int
        Number of points written.
    """
    sweep = load_radar_csv(input_path, config)
    x, y = polar_to_cartesian(sweep.angles_rad, sweep.ranges)
    intensities = sweep.intensities

    mask = intensities > threshold
    out_df = pd.DataFrame({
        "x": x[mask],
        "y": y[mask],
        "z": intensities[mask],
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    return len(out_df)


def aligned_inputs(
    base_dir: Path,
    gains: Tuple[int, ...],
) -> Iterable[Tuple[int, Dict[int, Path]]]:
    """
    Yield aligned sets of CSV files across gain folders.

    Parameters
    ----------
    base_dir : Path
        Directory containing gain_* subdirectories.
    gains : tuple
        Gain values to align.

    Yields
    ------
    tuple
        (index, {gain: path}) pairs.
    """
    lists: Dict[int, list] = {}
    for g in gains:
        folder = base_dir / f"gain_{g}"
        files = sorted(folder.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSVs found in {folder}")
        lists[g] = files

    count = min(len(v) for v in lists.values())
    for idx in range(count):
        yield idx + 1, {g: lists[g][idx] for g in gains}


def convert_batch_aligned(
    base_dir: Path,
    output_dir: Path,
    gains: Optional[Tuple[int, ...]] = None,
    threshold: float = 0.0,
    limit: Optional[int] = None,
    config: Optional[RadarConfig] = None,
) -> None:
    """
    Batch convert aligned gain sweeps to Cartesian CSVs.

    Parameters
    ----------
    base_dir : Path
        Directory containing gain_* subdirectories.
    output_dir : Path
        Output directory for Cartesian CSVs.
    gains : tuple, optional
        Gain values to process.
    threshold : float
        Minimum intensity threshold.
    limit : int, optional
        Maximum number of aligned sets to process.
    config : RadarConfig, optional
        Radar configuration.
    """
    if gains is None:
        gains = GainConfig().values

    for idx, group in islice(aligned_inputs(base_dir, gains), limit):
        for gain, src in group.items():
            out_name = f"{idx:04d}_gain_{gain}_cartesian.csv"
            out_path = output_dir / f"gain_{gain}" / out_name
            n_points = convert_single_csv(src, out_path, threshold, config)
            print(f"[{idx:04d}] gain {gain}: {src.name} -> {out_path} ({n_points:,} points)")

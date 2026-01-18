#!/usr/bin/env python3
"""
Batch-convert aligned gain sweeps to Cartesian CSVs.

- Finds matching rows across gain_40 / gain_50 / gain_75 (aligned by sort order).
- Writes outputs to gain-sorted folders with a shared index (0001, 0002, ...).
"""
# run with: cd "PointCloudWork" python 1_batch_csv_to_cartesian.py --base-input "../(.125NM)data_pattern3(.125NM)" --output-dir "./1.5_Folder" --threshold 0 --limit 5

from __future__ import annotations
import argparse
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

GAINS = (40, 50, 75)  # edit if you add/remove gains


def load_radar_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_names = ["Status", "Scale", "Range", "Gain",
                 "Angle"] + [f"Echo_{i}" for i in range(1024)]
    df = pd.read_csv(path, header=None, names=col_names,
                     skiprows=1, engine="c")
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    angles_rad = np.deg2rad(df["Angle"].to_numpy(
        np.float32) * (360.0 / 8196.0))
    echo_data = df.iloc[:, 5:].fillna(0).to_numpy(np.float32)
    max_ranges = df["Scale"].to_numpy(np.float32)
    num_bins = echo_data.shape[1]
    ranges = (max_ranges[:, None] / num_bins) * \
        np.arange(num_bins, dtype=np.float32)
    return angles_rad, ranges, echo_data


def to_cartesian(angles_rad: np.ndarray, ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = ranges * np.cos(angles_rad[:, None])
    y = ranges * np.sin(angles_rad[:, None])
    return x, y


def aligned_inputs(base_dir: Path, gains: Iterable[int]) -> Iterable[tuple[int, Dict[int, Path]]]:
    lists: Dict[int, list[Path]] = {}
    for g in gains:
        folder = base_dir / f"gain_{g}"
        files = sorted(folder.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSVs found in {folder}")
        lists[g] = files
    count = min(len(v) for v in lists.values())
    for idx in range(count):
        yield idx + 1, {g: lists[g][idx] for g in gains}


def convert_one(src: Path, dst: Path, threshold: float) -> int:
    angles_rad, ranges, intensities = load_radar_csv(src)
    x, y = to_cartesian(angles_rad, ranges)
    mask = intensities > threshold
    out_df = pd.DataFrame({"x": x[mask], "y": y[mask], "z": intensities[mask]})
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dst, index=False)
    return len(out_df)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_base = script_dir.parent / "(.125NM)data_pattern3(.125NM)"
    default_out = script_dir / "1.5_Folder"

    p = argparse.ArgumentParser(
        description="Batch convert gain sweeps to Cartesian CSVs.")
    p.add_argument("--base-input", type=Path, default=default_base,
                   help=f"Folder containing gain_*/ CSVs (default: {default_base})")
    p.add_argument("--output-dir", type=Path, default=default_out,
                   help=f"Where to write gain-sorted Cartesian CSVs (default: {default_out})")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Drop points with intensity <= threshold.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N aligned sets.")
    args = p.parse_args()

    for idx, group in islice(aligned_inputs(args.base_input, GAINS), args.limit):
        for gain, src in group.items():
            out_name = f"{idx:04d}_gain_{gain}_cartesian.csv"
            out_path = args.output_dir / f"gain_{gain}" / out_name
            n_points = convert_one(src, out_path, args.threshold)
            print(
                f"[{idx:04d}] gain {gain}: {src.name} -> {out_path} ({n_points:,} points)")


if __name__ == "__main__":
    main()

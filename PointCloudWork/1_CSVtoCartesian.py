#!/usr/bin/env python3
"""
Convert a radar sweep CSV to Cartesian point grid:
- x, y: horizontal distances (meters)
- z: intensity (echo value)

Usage:
python 1CSVtoCartesian.py [output.csv] [--input INPUT] [--threshold THRESH]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_radar_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Columns: Status, Scale, Range, Gain, Angle, Echo_0 ... Echo_1023
    col_names = ["Status", "Scale", "Range", "Gain",
                 "Angle"] + [f"Echo_{i}" for i in range(1024)]
    df = pd.read_csv(path, header=None, names=col_names,
                     skiprows=1, engine="c")
    if df.empty:
        raise ValueError("CSV is empty")
    angles_rad = np.deg2rad(df["Angle"].to_numpy(
        np.float32) * (360.0 / 8196.0))
    echo_data = df.iloc[:, 5:].fillna(0).to_numpy(
        np.float32)  # shape: (num_angles, num_bins)
    max_ranges = df["Scale"].to_numpy(
        np.float32)              # shape: (num_angles,)
    num_bins = echo_data.shape[1]
    # per-angle range resolution
    range_res = max_ranges[:, None] / num_bins
    # shape: (num_angles, num_bins)
    ranges = range_res * np.arange(num_bins, dtype=np.float32)
    return angles_rad, ranges, echo_data


def to_cartesian(angles_rad: np.ndarray, ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = ranges * np.cos(angles_rad[:, None])
    y = ranges * np.sin(angles_rad[:, None])
    return x, y


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / \
        "gain_75.csv"
    default_output = script_dir / "1.5_Folder" / "1.5_TEST_gain_75.csv"

    p = argparse.ArgumentParser(
        description="Convert radar sweep CSV to Cartesian point grid.")
    p.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=default_output,
        help=f"Path to write flat CSV of x,y,z points (default: {default_output}).",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Path to radar CSV (default: {default_input})",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Drop points with intensity <= threshold.",
    )
    args = p.parse_args()

    angles_rad, ranges, intensities = load_radar_csv(args.input)
    x, y = to_cartesian(angles_rad, ranges)

    mask = intensities > args.threshold
    x_flat = x[mask]
    y_flat = y[mask]
    z_flat = intensities[mask]

    out_df = pd.DataFrame({"x": x_flat, "y": y_flat, "z": z_flat})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df):,} points to {args.output}")


if __name__ == "__main__":
    main()

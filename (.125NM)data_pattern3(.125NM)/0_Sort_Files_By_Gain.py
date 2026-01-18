#!/usr/bin/env python3
"""
Sort radar sweep CSVs into gain-based folders (gain_40, gain_50, gain_75).

Usage (from repo root):
    python sort_by_gain.py
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent
TARGETS = {40: "gain_40", 50: "gain_50", 75: "gain_75"}


def sniff_gain(csv_path: Path) -> Optional[int]:
    """Read the first data row and return the Gain column as int; None if missing/invalid."""
    with csv_path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)  # skip header
            row = next(reader)
        except StopIteration:
            return None
    if len(row) < 4:  # need at least up to Gain column
        return None
    try:
        return int(float(row[3]))  # Gain is the 4th column (index 3)
    except ValueError:
        return None


def ensure_dirs(base: Path) -> None:
    for name in TARGETS.values():
        (base / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Folder not found: {DATA_DIR}")
    ensure_dirs(DATA_DIR)

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    for path in csv_files:
        gain = sniff_gain(path)
        target_name = TARGETS.get(gain)
        if target_name is None:
            print(f"skip (unknown gain): {path.name}")
            continue
        dest = DATA_DIR / target_name / path.name
        print(f"moving gain {gain}: {path.name} -> {target_name}/")
        path.rename(dest)


if __name__ == "__main__":
    main()

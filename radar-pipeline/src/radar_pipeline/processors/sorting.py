"""File sorting by radar gain value."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import GainConfig


def sniff_gain(csv_path: Path) -> Optional[int]:
    """
    Read the first data row and return the Gain column value.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file.

    Returns
    -------
    int or None
        Gain value, or None if missing/invalid.
    """
    with csv_path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)  # skip header
            row = next(reader)
        except StopIteration:
            return None

    if len(row) < 4:  # need at least up to Gain column
        return None

    try:
        return int(float(row[3]))  # Gain is the 4th column (index 3)
    except ValueError:
        return None


def sort_files_by_gain(
    source_dir: Path,
    gains: Optional[Tuple[int, ...]] = None,
) -> Dict[int, List[Path]]:
    """
    Sort CSV files by gain value without moving them.

    Parameters
    ----------
    source_dir : Path
        Directory containing CSV files.
    gains : tuple, optional
        Gain values to look for. Defaults to (40, 50, 75).

    Returns
    -------
    dict
        Mapping of gain value to list of file paths.
    """
    if gains is None:
        gains = GainConfig().values

    result: Dict[int, List[Path]] = {g: [] for g in gains}

    for csv_path in sorted(source_dir.glob("*.csv")):
        gain = sniff_gain(csv_path)
        if gain in result:
            result[gain].append(csv_path)

    return result


def move_files_to_gain_folders(
    source_dir: Path,
    gains: Optional[Tuple[int, ...]] = None,
    dry_run: bool = False,
) -> Dict[int, List[Path]]:
    """
    Sort CSV files into gain-based subdirectories.

    Parameters
    ----------
    source_dir : Path
        Directory containing CSV files.
    gains : tuple, optional
        Gain values to create folders for. Defaults to (40, 50, 75).
    dry_run : bool
        If True, only report what would be moved.

    Returns
    -------
    dict
        Mapping of gain value to list of moved file paths.
    """
    if gains is None:
        gains = GainConfig().values

    targets = {g: f"gain_{g}" for g in gains}
    moved: Dict[int, List[Path]] = {g: [] for g in gains}

    # Ensure target directories exist
    if not dry_run:
        for name in targets.values():
            (source_dir / name).mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(source_dir.glob("*.csv")):
        gain = sniff_gain(csv_path)
        target_name = targets.get(gain)

        if target_name is None:
            continue

        dest = source_dir / target_name / csv_path.name

        if dry_run:
            print(f"Would move gain {gain}: {csv_path.name} -> {target_name}/")
        else:
            csv_path.rename(dest)
            print(f"Moved gain {gain}: {csv_path.name} -> {target_name}/")

        moved[gain].append(dest if not dry_run else csv_path)

    return moved

"""File filtering by radar range value."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Set

from ..config import GainConfig


def get_csv_range(path: Path) -> Optional[int]:
    """
    Read the Range column from the first data row.

    Parameters
    ----------
    path : Path
        Path to CSV file.

    Returns
    -------
    int or None
        Range value, or None if missing/invalid.
    """
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)  # header
            row = next(reader)
        except StopIteration:
            return None

    if len(row) < 3:
        return None

    try:
        return int(float(row[2]))  # Range is the 3rd column (index 2)
    except ValueError:
        return None


def find_targets(
    base_dir: Path,
    gains: Optional[tuple[int, ...]] = None,
) -> Iterable[Path]:
    """
    Yield CSV paths in gain subdirectories.

    Parameters
    ----------
    base_dir : Path
        Base directory containing gain_* subdirectories.
    gains : tuple, optional
        Gain values to search in.

    Yields
    ------
    Path
        CSV file paths.
    """
    if gains is None:
        gains = GainConfig().values

    for g in gains:
        folder = base_dir / f"gain_{g}"
        if not folder.is_dir():
            continue
        yield from folder.glob("*.csv")


def find_files_by_range(
    base_dir: Path,
    ranges_to_find: Set[int],
    gains: Optional[tuple[int, ...]] = None,
) -> List[Path]:
    """
    Find CSV files with specific Range values.

    Parameters
    ----------
    base_dir : Path
        Base directory containing gain subdirectories.
    ranges_to_find : set
        Range values to match.
    gains : tuple, optional
        Gain values to search in.

    Returns
    -------
    list
        Paths to matching CSV files.
    """
    matches = []
    for path in find_targets(base_dir, gains):
        rng = get_csv_range(path)
        if rng in ranges_to_find:
            matches.append(path)
    return matches


def remove_files_by_range(
    base_dir: Path,
    ranges_to_remove: Set[int],
    gains: Optional[tuple[int, ...]] = None,
    dry_run: bool = False,
) -> List[Path]:
    """
    Delete CSV files with specific Range values.

    Parameters
    ----------
    base_dir : Path
        Base directory containing gain subdirectories.
    ranges_to_remove : set
        Range values to delete.
    gains : tuple, optional
        Gain values to search in.
    dry_run : bool
        If True, only report what would be deleted.

    Returns
    -------
    list
        Paths to deleted (or would-be-deleted) files.
    """
    to_delete = find_files_by_range(base_dir, ranges_to_remove, gains)

    if not to_delete:
        print(f"No files with Range in {ranges_to_remove} found.")
        return []

    action = "Would delete" if dry_run else "Deleting"
    print(f"{action} {len(to_delete)} files:")

    for path in to_delete:
        print(f"  - {path}")
        if not dry_run:
            path.unlink(missing_ok=True)

    return to_delete

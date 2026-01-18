#!/usr/bin/env python3
"""
Delete CSV sweep files whose Range column is 1 or 2 from gain folders.

Targets the sibling directory "(.125NM)data_pattern3(.125NM)" and the
subfolders gain_40, gain_50, gain_75.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def find_targets(base_dir: Path) -> Iterable[Path]:
    """Yield CSV paths in gain_40/gain_50/gain_75 under base_dir."""
    for sub in ("gain_40", "gain_50", "gain_75"):
        folder = base_dir / sub
        if not folder.is_dir():
            continue
        yield from folder.glob("*.csv")


def should_delete(path: Path) -> bool:
    """Return True if the first data row has Range == 1 or 2 (index 2)."""
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)  # header
            row = next(reader)
        except StopIteration:
            return False
    if len(row) < 3:
        return False
    try:
        rng = int(float(row[2]))
    except ValueError:
        return False
    return rng in (1, 2)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_base = script_dir.parent / "(.125NM)data_pattern3(.125NM)"

    p = argparse.ArgumentParser(description="Remove CSV sweeps with Range 1 or 2.")
    p.add_argument(
        "--base",
        type=Path,
        default=default_base,
        help=f"Base directory containing gain_40/50/75 (default: {default_base})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be removed without deleting.",
    )
    args = p.parse_args()

    if not args.base.exists():
        raise SystemExit(f"Base directory not found: {args.base}")

    to_delete = [p for p in find_targets(args.base) if should_delete(p)]
    if not to_delete:
        print("No files with Range 1 or 2 found.")
        return

    action = "Would delete" if args.dry_run else "Deleting"
    print(f"{action} {len(to_delete)} files:")
    for path in to_delete:
        print(f" - {path}")
        if not args.dry_run:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

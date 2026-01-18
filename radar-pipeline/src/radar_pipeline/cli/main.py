"""Main CLI entry point for radar pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from .. import __version__
from ..config import PipelineConfig


pass_config = click.make_pass_decorator(PipelineConfig, ensure=True)


@click.group()
@click.option(
    "-c", "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML config file.",
)
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: int) -> None:
    """Radar point cloud processing pipeline."""
    ctx.ensure_object(dict)

    if config:
        ctx.obj["config"] = PipelineConfig.from_yaml(config)
    else:
        ctx.obj["config"] = PipelineConfig()

    ctx.obj["verbose"] = verbose


@cli.command("sort-by-gain")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Show what would be moved without moving.")
@click.pass_context
def sort_by_gain(ctx: click.Context, directory: Path, dry_run: bool) -> None:
    """Sort CSV files into gain_40/50/75 folders."""
    from ..processors.sorting import move_files_to_gain_folders

    config: PipelineConfig = ctx.obj["config"]
    moved = move_files_to_gain_folders(directory, config.gains.values, dry_run=dry_run)

    total = sum(len(v) for v in moved.values())
    action = "Would move" if dry_run else "Moved"
    click.echo(f"{action} {total} files total.")


@cli.command("filter-range")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--ranges", "-r",
    multiple=True,
    type=int,
    default=[1, 2],
    help="Range values to remove.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting.")
@click.pass_context
def filter_range(ctx: click.Context, directory: Path, ranges: tuple, dry_run: bool) -> None:
    """Remove CSV files with specified Range values."""
    from ..processors.filtering import remove_files_by_range

    config: PipelineConfig = ctx.obj["config"]
    removed = remove_files_by_range(
        directory,
        set(ranges),
        config.gains.values,
        dry_run=dry_run,
    )

    action = "Would remove" if dry_run else "Removed"
    click.echo(f"{action} {len(removed)} files.")


@cli.command("convert")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--threshold", "-t", type=float, default=0.0, help="Intensity threshold.")
@click.option("--batch/--single", default=False, help="Batch mode for aligned gains.")
@click.option("--limit", type=int, help="Limit number of files in batch mode.")
@click.pass_context
def convert(
    ctx: click.Context,
    input_path: Path,
    output_path: Path,
    threshold: float,
    batch: bool,
    limit: Optional[int],
) -> None:
    """Convert radar CSV to Cartesian coordinates."""
    config: PipelineConfig = ctx.obj["config"]

    if batch:
        from ..processors.cartesian import convert_batch_aligned

        convert_batch_aligned(
            input_path,
            output_path,
            config.gains.values,
            threshold,
            limit,
        )
        click.echo("Batch conversion complete.")
    else:
        from ..processors.cartesian import convert_single_csv

        n_points = convert_single_csv(input_path, output_path, threshold)
        click.echo(f"Saved {n_points:,} points to {output_path}")


@cli.command("build")
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--flat/--no-flat", default=True, help="Generate flat stack.")
@click.option("--offset/--no-offset", default=True, help="Generate offset stack.")
@click.option("--plot/--no-plot", default=True, help="Generate PNG previews.")
@click.pass_context
def build(
    ctx: click.Context,
    input_dir: Path,
    output_dir: Path,
    flat: bool,
    offset: bool,
    plot: bool,
) -> None:
    """Build stacked PLY point clouds from Cartesian CSVs."""
    from ..processors.point_cloud import build_stacked_clouds
    from ..visualization.plotting import plot_point_cloud
    from ..core.loaders import load_ply

    config: PipelineConfig = ctx.obj["config"]

    outputs = build_stacked_clouds(
        input_dir,
        output_dir,
        config.processing,
        config.gains,
        config.radar,
        generate_flat=flat,
        generate_offset=offset,
    )

    if plot:
        for name, ply_path in outputs.items():
            png_path = ply_path.with_suffix(".png")
            cloud = load_ply(ply_path)
            plot_point_cloud(
                png_path,
                cloud,
                title=f"{ply_path.stem}",
                max_points=config.processing.plot_max_points,
            )
            click.echo(f"Plot saved: {png_path.name}")

    click.echo("Build complete.")


@cli.command("visualize")
@click.argument("ply_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output PNG path.",
)
@click.option("--max-points", type=int, default=1_000_000, help="Maximum points to plot.")
@click.option("--alpha", type=float, default=0.5, help="Marker opacity.")
@click.pass_context
def visualize(
    ctx: click.Context,
    ply_file: Path,
    output: Optional[Path],
    max_points: int,
    alpha: float,
) -> None:
    """Generate PNG preview from PLY point cloud."""
    from ..visualization.plotting import plot_ply_preview

    output_path = plot_ply_preview(ply_file, output, max_points, alpha)
    click.echo(f"Preview saved to {output_path}")


@cli.command("cluster")
@click.argument("ply_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    help="Output directory.",
)
@click.option("--eps-space", type=float, help="Spatial epsilon.")
@click.option("--eps-time", type=float, help="Temporal epsilon.")
@click.option("--min-samples", type=int, help="Minimum samples per cluster.")
@click.option("--max-points", type=int, help="Maximum points to process.")
@click.option("--plot/--no-plot", default=True, help="Generate PNG visualization.")
@click.pass_context
def cluster(
    ctx: click.Context,
    ply_file: Path,
    output_dir: Optional[Path],
    eps_space: Optional[float],
    eps_time: Optional[float],
    min_samples: Optional[int],
    max_points: Optional[int],
    plot: bool,
) -> None:
    """Run ST-DBSCAN clustering on point cloud."""
    from ..processors.clustering import process_ply_clustering
    from ..core.loaders import load_ply
    from ..visualization.plotting import plot_labeled_cloud
    import numpy as np

    config: PipelineConfig = ctx.obj["config"]

    # Override config with CLI options
    cluster_config = config.clustering.model_copy()
    if eps_space is not None:
        cluster_config.eps_space = eps_space
    if eps_time is not None:
        cluster_config.eps_time = eps_time
    if min_samples is not None:
        cluster_config.min_samples = min_samples
    if max_points is not None:
        cluster_config.max_points = max_points

    if output_dir is None:
        output_dir = ply_file.parent

    csv_path, labels = process_ply_clustering(
        ply_file,
        output_dir,
        cluster_config,
        config.gains,
    )

    if plot:
        cloud = load_ply(ply_file)
        png_path = output_dir / f"{ply_file.stem}_dbscan_labels.png"
        plot_labeled_cloud(
            png_path,
            cloud.to_coords(),
            labels,
            cloud.colors,
            title=f"ST-DBSCAN: {ply_file.name}",
            max_points=config.processing.plot_max_points,
        )
        click.echo(f"Plot saved: {png_path}")

    click.echo(f"Clustering complete. Labels saved to {csv_path}")


if __name__ == "__main__":
    cli()

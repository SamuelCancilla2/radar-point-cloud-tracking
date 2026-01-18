"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner
from pathlib import Path

from radar_pipeline.cli.main import cli


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


def test_cli_help(runner: CliRunner):
    """Test CLI help command."""
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Radar point cloud processing pipeline" in result.output


def test_cli_version(runner: CliRunner):
    """Test CLI version command."""
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_sort_by_gain_help(runner: CliRunner):
    """Test sort-by-gain help."""
    result = runner.invoke(cli, ["sort-by-gain", "--help"])

    assert result.exit_code == 0
    assert "Sort CSV files into gain" in result.output


def test_filter_range_help(runner: CliRunner):
    """Test filter-range help."""
    result = runner.invoke(cli, ["filter-range", "--help"])

    assert result.exit_code == 0
    assert "Remove CSV files" in result.output


def test_convert_help(runner: CliRunner):
    """Test convert help."""
    result = runner.invoke(cli, ["convert", "--help"])

    assert result.exit_code == 0
    assert "Cartesian coordinates" in result.output


def test_build_help(runner: CliRunner):
    """Test build help."""
    result = runner.invoke(cli, ["build", "--help"])

    assert result.exit_code == 0
    assert "stacked PLY" in result.output


def test_visualize_help(runner: CliRunner):
    """Test visualize help."""
    result = runner.invoke(cli, ["visualize", "--help"])

    assert result.exit_code == 0
    assert "PNG preview" in result.output


def test_cluster_help(runner: CliRunner):
    """Test cluster help."""
    result = runner.invoke(cli, ["cluster", "--help"])

    assert result.exit_code == 0
    assert "ST-DBSCAN" in result.output


def test_sort_by_gain_dry_run(runner: CliRunner, tmp_path: Path):
    """Test sort-by-gain with dry-run."""
    # Create test directory with CSV
    csv_content = "Status,Scale,Range,Gain,Angle\n1,496,3,75,10"
    (tmp_path / "test.csv").write_text(csv_content)

    result = runner.invoke(cli, ["sort-by-gain", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0

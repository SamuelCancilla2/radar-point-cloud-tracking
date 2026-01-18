#!/usr/bin/env python3
"""
Run ST-DBSCAN experiments with different parameter settings.
Tracks timing and generates a comparison report.

QUICK START: Just press Run! Default settings are lightweight.
To customize, change the settings below.
"""

import subprocess
import time
import json
import gc
import sys
from pathlib import Path
from datetime import datetime
import shutil

# =============================================================================
# EASY CONFIGURATION - Change these values to customize your run
# =============================================================================

# Number of frames to process (lower = faster, use 5-10 for quick tests)
MAX_FRAMES = 10

# Set to True for a quick single test, False to run all experiments
QUICK_MODE = True

# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

# Quick mode runs just the default experiment
QUICK_EXPERIMENT = {
    "name": "default",
    "eps_space": 8.0,
    "eps_time": 2.0,
    "min_samples": 15,
    "min_frames": 2,
    "description": "Default parameters"
}

# Full experiment list (used when QUICK_MODE = False)
FULL_EXPERIMENTS = [
    {
        "name": "default",
        "eps_space": 8.0,
        "eps_time": 2.0,
        "min_samples": 15,
        "min_frames": 2,
        "description": "Default parameters"
    },
    {
        "name": "tight_spatial",
        "eps_space": 5.0,
        "eps_time": 2.0,
        "min_samples": 15,
        "min_frames": 2,
        "description": "Tighter spatial radius (5m)"
    },
    {
        "name": "aggressive",
        "eps_space": 5.0,
        "eps_time": 1.5,
        "min_samples": 25,
        "min_frames": 3,
        "description": "Aggressive filtering"
    },
]

# Select which experiments to run based on mode
EXPERIMENTS = [QUICK_EXPERIMENT] if QUICK_MODE else FULL_EXPERIMENTS

def run_experiment(exp: dict, script_dir: Path, max_frames: int = 0) -> dict:
    """Run a single experiment and return results."""

    output_name = f"results_{exp['name']}_epsS{exp['eps_space']}_epsT{exp['eps_time']}_minS{exp['min_samples']}_minF{exp['min_frames']}"
    output_dir = script_dir / output_name

    # Build command - use low memory settings to avoid WSL crashes
    cmd = [
        sys.executable, str(script_dir / "stdbscan_denoising_pipeline.py"),
        "--eps-space", str(exp["eps_space"]),
        "--eps-time", str(exp["eps_time"]),
        "--min-samples", str(exp["min_samples"]),
        "--min-frames", str(exp["min_frames"]),
        "--max-frames", str(max_frames),
        "--output-dir", str(output_dir),
        "--no-parallel",  # More stable, uses less memory
        "--low-memory",   # Free intermediate data to reduce RAM
        "--no-viz",       # Skip visualization to save RAM
    ]

    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"Parameters: eps_space={exp['eps_space']}, eps_time={exp['eps_time']}, "
          f"min_samples={exp['min_samples']}, min_frames={exp['min_frames']}")
    print(f"Output: {output_dir.name}")
    print('='*60)

    # Time the run
    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 600s")
        result = type('obj', (object,), {'stdout': '', 'stderr': 'TIMEOUT'})()
    end_time = time.time()

    elapsed = end_time - start_time

    # Force garbage collection to free memory
    gc.collect()

    # Parse output for statistics
    output = result.stdout + result.stderr
    print(output)

    # Extract stats from output
    stats = {
        "name": exp["name"],
        "description": exp["description"],
        "eps_space": exp["eps_space"],
        "eps_time": exp["eps_time"],
        "min_samples": exp["min_samples"],
        "min_frames": exp["min_frames"],
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        "total_points": 0,
        "noise_points": 0,
        "signal_points": 0,
        "num_clusters": 0,
        "noise_pct": 0.0,
    }

    # Try to read stats from CSV
    stats_file = output_dir / "denoising_stats.csv"
    if stats_file.exists():
        import pandas as pd
        df = pd.read_csv(stats_file)
        if len(df) > 0:
            row = df.iloc[0]
            stats["total_points"] = int(row.get("total_points", 0))
            stats["noise_points"] = int(row.get("noise_points", 0))
            stats["signal_points"] = int(row.get("signal_points", 0))
            stats["num_clusters"] = int(row.get("num_clusters", 0))
            stats["noise_pct"] = float(row.get("noise_reduction_pct", 0))

    return stats


def generate_latex_report(results: list, output_path: Path):
    """Generate LaTeX report from experiment results."""

    latex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{longtable}

\title{ST-DBSCAN Radar Point Cloud Denoising\\Parameter Comparison Report}
\author{Generated by Automated Pipeline}
\date{""" + datetime.now().strftime("%B %d, %Y") + r"""}

\begin{document}
\maketitle

\section{Introduction}
This report compares different parameter settings for the ST-DBSCAN (Spatio-Temporal DBSCAN)
radar point cloud denoising pipeline. The goal is to identify optimal parameters for
filtering transient noise while preserving persistent radar returns.

\subsection{Parameters Tested}
\begin{itemize}
    \item \textbf{eps\_space}: Spatial clustering radius in meters
    \item \textbf{eps\_time}: Temporal clustering window in frames
    \item \textbf{min\_samples}: Minimum points required to form a cluster
    \item \textbf{min\_frames}: Minimum number of frames a cluster must span
\end{itemize}

\section{Experiment Results}

\begin{table}[H]
\centering
\caption{Summary of Experiment Results}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Experiment} & \textbf{Noise \%} & \textbf{Clusters} & \textbf{Signal Pts} & \textbf{Time (s)} \\
\midrule
"""

    for r in results:
        latex += f"{r['name'].replace('_', '\\_')} & {r['noise_pct']:.1f}\\% & {r['num_clusters']} & {r['signal_points']:,} & {r['elapsed_seconds']:.1f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}

\section{Detailed Parameter Settings}

\begin{longtable}{lp{2cm}p{1.5cm}p{1.5cm}p{1.5cm}p{4cm}}
\toprule
\textbf{Name} & \textbf{eps\_space} & \textbf{eps\_time} & \textbf{min\_samples} & \textbf{min\_frames} & \textbf{Description} \\
\midrule
\endhead
"""

    for r in results:
        latex += f"{r['name'].replace('_', '\\_')} & {r['eps_space']} & {r['eps_time']} & {r['min_samples']} & {r['min_frames']} & {r['description']} \\\\\n"

    latex += r"""\bottomrule
\end{longtable}

\section{Analysis}

\subsection{Noise Reduction Comparison}
"""

    # Find best and worst
    best_noise = max(results, key=lambda x: x['noise_pct'])
    least_noise = min(results, key=lambda x: x['noise_pct'])
    most_clusters = max(results, key=lambda x: x['num_clusters'])
    fastest = min(results, key=lambda x: x['elapsed_seconds'])

    latex += f"""
The experiment with the highest noise reduction was \\textbf{{{best_noise['name'].replace('_', '\\_')}}}
with {best_noise['noise_pct']:.1f}\\% of points classified as noise.

The experiment with the lowest noise reduction was \\textbf{{{least_noise['name'].replace('_', '\\_')}}}
with {least_noise['noise_pct']:.1f}\\% noise.

The configuration that detected the most clusters was \\textbf{{{most_clusters['name'].replace('_', '\\_')}}}
with {most_clusters['num_clusters']} clusters identified.

The fastest experiment was \\textbf{{{fastest['name'].replace('_', '\\_')}}}
completing in {fastest['elapsed_seconds']:.1f} seconds.

\subsection{{Recommendations}}
Based on these results:
\\begin{{itemize}}
    \\item For \\textbf{{maximum noise removal}}: Use the ``{best_noise['name'].replace('_', '\\_')}'' configuration
    \\item For \\textbf{{preserving more objects}}: Use the ``{least_noise['name'].replace('_', '\\_')}'' configuration
    \\item For \\textbf{{fastest processing}}: Use the ``{fastest['name'].replace('_', '\\_')}'' configuration
\\end{{itemize}}

\\section{{Raw Data}}

\\begin{{verbatim}}
"""

    # Add raw JSON data
    for r in results:
        latex += f"{r['name']}:\n"
        latex += f"  Total points: {r['total_points']:,}\n"
        latex += f"  Noise points: {r['noise_points']:,} ({r['noise_pct']:.2f}%)\n"
        latex += f"  Signal points: {r['signal_points']:,}\n"
        latex += f"  Clusters: {r['num_clusters']}\n"
        latex += f"  Time: {r['elapsed_seconds']:.2f}s\n\n"

    latex += r"""\end{verbatim}

\end{document}
"""

    output_path.write_text(latex)
    print(f"LaTeX report written to: {output_path}")


def main():
    script_dir = Path(__file__).resolve().parent

    print("="*60)
    print("ST-DBSCAN PARAMETER COMPARISON EXPERIMENTS")
    print("="*60)
    print(f"Mode: {'QUICK (single experiment)' if QUICK_MODE else 'FULL (all experiments)'}")
    print(f"Running {len(EXPERIMENTS)} experiment(s) on {MAX_FRAMES} frames each")
    print("="*60)

    results = []
    total_start = time.time()

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n[Experiment {i+1}/{len(EXPERIMENTS)}]")
        stats = run_experiment(exp, script_dir, max_frames=MAX_FRAMES)
        results.append(stats)

        # Save intermediate results
        with open(script_dir / "experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)

    total_elapsed = time.time() - total_start

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print("="*60)

    # Generate LaTeX report
    report_path = script_dir / "stdbscan_comparison_report.tex"
    generate_latex_report(results, report_path)

    # Save JSON results
    with open(script_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - {report_path}")
    print(f"  - {script_dir / 'experiment_results.json'}")


if __name__ == "__main__":
    main()

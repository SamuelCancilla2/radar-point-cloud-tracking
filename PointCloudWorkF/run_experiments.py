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
# DEPENDENCY CHECK - Install missing packages automatically
# =============================================================================

def check_and_install_dependencies():
    """Check for required packages and install if missing."""
    required = ['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'pillow']
    missing = []

    for package in required:
        try:
            if package == 'pillow':
                __import__('PIL')
            elif package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet'] + missing)
        print("Dependencies installed successfully!\n")

# Run dependency check on import
check_and_install_dependencies()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# =============================================================================
# EASY CONFIGURATION - Change these values to customize your run
# =============================================================================

# Number of frames to process (lower = faster, use 5-10 for quick tests)
MAX_FRAMES = 10

# Set to True for a quick single test, False to run all experiments
QUICK_MODE = False

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
        "--skip-gif",     # Skip slow GIF generation, keep fast PNGs
    ]

    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"Parameters: eps_space={exp['eps_space']}, eps_time={exp['eps_time']}, "
          f"min_samples={exp['min_samples']}, min_frames={exp['min_frames']}")
    print(f"Output: {output_dir.name}")
    print('='*60)
    sys.stdout.flush()  # Ensure output is displayed immediately

    # Time the run - stream output live so user sees progress
    start_time = time.time()
    try:
        # Stream output live instead of capturing
        process = subprocess.Popen(
            cmd, cwd=script_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        output_lines = []
        for line in process.stdout:
            print(f"  {line}", end='')  # Print live with indent
            sys.stdout.flush()
            output_lines.append(line)
        process.wait(timeout=600)
        output = ''.join(output_lines)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 600s")
        process.kill()
        output = 'TIMEOUT'
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"\n  Completed in {elapsed:.1f} seconds")

    # Force garbage collection to free memory
    gc.collect()

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

\\subsection{{Recommendations}}
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


def generate_summary_pngs(results: list, script_dir: Path):
    """Generate summary PNG visualizations comparing all experiments."""
    if not HAS_MPL:
        print("matplotlib not available, skipping summary visualizations")
        return

    print("\n" + "="*60)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("="*60)
    print("Creating comparison charts...")
    sys.stdout.flush()

    # 1. Bar chart comparing noise reduction across experiments
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [r['name'].replace('_', '\n') for r in results]
    x = np.arange(len(names))

    # Noise reduction percentage
    ax1 = axes[0]
    noise_pcts = [r['noise_pct'] for r in results]
    bars1 = ax1.bar(x, noise_pcts, color='#e74c3c', edgecolor='black')
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Noise Removed (%)')
    ax1.set_title('Noise Reduction by Parameter Set')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    for bar, val in zip(bars1, noise_pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Number of clusters found
    ax2 = axes[1]
    clusters = [r['num_clusters'] for r in results]
    bars2 = ax2.bar(x, clusters, color='#9b59b6', edgecolor='black')
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Number of Clusters')
    ax2.set_title('Clusters Detected by Parameter Set')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    for bar, val in zip(bars2, clusters):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val}', ha='center', va='bottom', fontsize=10)

    # Signal points retained
    ax3 = axes[2]
    signal_pts = [r['signal_points'] for r in results]
    bars3 = ax3.bar(x, signal_pts, color='#2ecc71', edgecolor='black')
    ax3.set_xlabel('Experiment')
    ax3.set_ylabel('Signal Points Retained')
    ax3.set_title('Signal Points by Parameter Set')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=9)
    for bar, val in zip(bars3, signal_pts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(signal_pts)*0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    comparison_path = script_dir / "experiment_comparison.png"
    plt.savefig(comparison_path, dpi=200)
    plt.close()
    print(f"Saved: {comparison_path}")

    # 2. Parameter vs Results scatter/bubble chart
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))

        eps_space = [r['eps_space'] for r in results]
        min_samples = [r['min_samples'] for r in results]
        noise_pcts = [r['noise_pct'] for r in results]
        clusters = [r['num_clusters'] for r in results]

        # Bubble size based on clusters, color based on noise reduction
        scatter = ax.scatter(eps_space, min_samples,
                           s=[c*50 + 100 for c in clusters],  # size by clusters
                           c=noise_pcts, cmap='RdYlGn_r',  # color by noise %
                           edgecolors='black', linewidth=1.5, alpha=0.7)

        # Add labels for each point
        for i, r in enumerate(results):
            ax.annotate(r['name'], (eps_space[i], min_samples[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('Spatial Radius (eps_space)', fontsize=11)
        ax.set_ylabel('Min Samples', fontsize=11)
        ax.set_title('Parameter Space Exploration\n(bubble size = clusters, color = noise %)', fontsize=12)
        cbar = plt.colorbar(scatter, ax=ax, label='Noise Removed (%)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        param_path = script_dir / "parameter_exploration.png"
        plt.savefig(param_path, dpi=200)
        plt.close()
        print(f"Saved: {param_path}")

    # 3. Summary table as image
    fig, ax = plt.subplots(figsize=(12, max(3, len(results) * 0.8 + 2)))
    ax.axis('off')

    table_data = []
    headers = ['Experiment', 'eps_space', 'eps_time', 'min_samples', 'min_frames',
               'Noise %', 'Clusters', 'Signal Pts', 'Time (s)']

    for r in results:
        table_data.append([
            r['name'],
            f"{r['eps_space']:.1f}",
            f"{r['eps_time']:.1f}",
            str(r['min_samples']),
            str(r['min_frames']),
            f"{r['noise_pct']:.1f}%",
            str(r['num_clusters']),
            f"{r['signal_points']:,}",
            f"{r['elapsed_seconds']:.1f}"
        ])

    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                    cellLoc='center', colColours=['#3498db']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight best results
    if len(results) > 1:
        best_noise_idx = max(range(len(results)), key=lambda i: results[i]['noise_pct'])
        best_cluster_idx = max(range(len(results)), key=lambda i: results[i]['num_clusters'])
        table[(best_noise_idx + 1, 5)].set_facecolor('#c8e6c9')  # light green
        table[(best_cluster_idx + 1, 6)].set_facecolor('#c8e6c9')

    ax.set_title('Experiment Results Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    summary_path = script_dir / "results_summary_table.png"
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {summary_path}")

    print("Summary PNGs generated successfully!")


def main():
    script_dir = Path(__file__).resolve().parent

    print("="*60)
    print("ST-DBSCAN PARAMETER COMPARISON EXPERIMENTS")
    print("="*60)
    print(f"Mode: {'QUICK (single experiment)' if QUICK_MODE else 'FULL (all experiments)'}")
    print(f"Running {len(EXPERIMENTS)} experiment(s) on {MAX_FRAMES} frames each")
    print("="*60)
    print("\nExperiments to run:")
    for i, exp in enumerate(EXPERIMENTS):
        print(f"  {i+1}. {exp['name']}: eps_space={exp['eps_space']}, min_samples={exp['min_samples']}")
    print("\nStarting experiments...\n")
    sys.stdout.flush()

    results = []
    total_start = time.time()

    for i, exp in enumerate(EXPERIMENTS):
        remaining = len(EXPERIMENTS) - i
        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT {i+1}/{len(EXPERIMENTS)} - {remaining} remaining")
        print(f"{'#'*60}")
        sys.stdout.flush()
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

    # Generate summary PNG visualizations
    generate_summary_pngs(results, script_dir)

    # Save JSON results
    with open(script_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - {report_path}")
    print(f"  - {script_dir / 'experiment_results.json'}")
    print(f"  - {script_dir / 'experiment_comparison.png'}")
    print(f"  - {script_dir / 'results_summary_table.png'}")
    if len(results) > 1:
        print(f"  - {script_dir / 'parameter_exploration.png'}")
    print(f"\nEach experiment folder also contains:")
    print(f"  - denoising_comparison.png (raw vs denoised)")
    print(f"  - noise_reduction_stats.png (pie/bar charts)")
    print(f"  - temporal_clusters.png (clusters across frames)")

    print("\n" + "="*60)
    print("SUCCESS! All experiments completed and visualizations generated.")
    print("="*60)


if __name__ == "__main__":
    main()

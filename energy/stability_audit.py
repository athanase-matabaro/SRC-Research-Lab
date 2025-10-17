#!/usr/bin/env python3
"""
Cross-Dataset Stability & Energy-Coherence Audit (Phase H.5.3)

Validates CAQ-E consistency across multiple datasets and compression scenarios.
Quantifies energy coherence and detects abnormal variance or drift.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Phase: H.5.3 - Cross-Dataset Stability & Energy-Coherence Audit
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats

# Optional plotting dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    matplotlib = None
    plt = None
    sns = None


def compute_stability_metrics(results: Dict) -> Dict:
    """
    Compute cross-dataset stability metrics.

    Args:
        results: Benchmark results dictionary with 'datasets' key

    Returns:
        Dictionary with stability metrics:
        - mean_caqe_var: Cross-dataset CAQ-E variance
        - energy_coherence: Energy-CAQ correlation
        - drift_index: Normalized drift from baseline
        - per_dataset_stats: Individual dataset statistics
    """
    datasets = results.get("datasets", {})

    if not datasets:
        raise ValueError("No datasets found in results")

    # Extract per-dataset CAQ-E values
    dataset_stats = {}

    for dataset_name, dataset_result in datasets.items():
        # Handle different result structures
        if "adaptive" in dataset_result:
            # Full benchmark structure
            adaptive = dataset_result["adaptive"]
            baseline = dataset_result.get("baseline", {})

            adaptive_caqe = adaptive.get("averages", {}).get("caq_e", 0.0)
            baseline_caqe = baseline.get("averages", {}).get("caq_e", 0.0)

            # Get all run CAQ-E values for variance
            adaptive_runs = adaptive.get("runs", [])
            caqe_values = [r.get("caq_e", 0.0) for r in adaptive_runs if "caq_e" in r]

            # Energy-CPU correlation from runs
            energies = [r.get("energy_joules", 0.0) for r in adaptive_runs if "energy_joules" in r]
            cpu_times = [r.get("cpu_seconds", 0.0) for r in adaptive_runs if "cpu_seconds" in r]
            ratios = [r.get("compression_ratio", 0.0) for r in adaptive_runs if "compression_ratio" in r]

        elif "runs" in dataset_result:
            # Direct runs structure
            runs = dataset_result["runs"]
            caqe_values = [r.get("caq_e", 0.0) for r in runs if "caq_e" in r]
            adaptive_caqe = np.mean(caqe_values) if caqe_values else 0.0
            baseline_caqe = 0.0

            energies = [r.get("energy_joules", 0.0) for r in runs if "energy_joules" in r]
            cpu_times = [r.get("cpu_seconds", 0.0) for r in runs if "cpu_seconds" in r]
            ratios = [r.get("compression_ratio", 0.0) for r in runs if "compression_ratio" in r]

        else:
            # Simple structure
            adaptive_caqe = dataset_result.get("caq_e", 0.0)
            baseline_caqe = 0.0
            caqe_values = [adaptive_caqe]
            energies = [dataset_result.get("energy_joules", 0.0)]
            cpu_times = [dataset_result.get("cpu_seconds", 0.0)]
            ratios = [dataset_result.get("compression_ratio", 0.0)]

        # Compute per-dataset statistics
        dataset_stats[dataset_name] = {
            "mean_caqe": float(adaptive_caqe),
            "baseline_caqe": float(baseline_caqe),
            "caqe_std": float(np.std(caqe_values)) if len(caqe_values) > 1 else 0.0,
            "caqe_values": caqe_values,
            "energies": energies,
            "cpu_times": cpu_times,
            "ratios": ratios,
        }

    # Compute cross-dataset metrics
    mean_caqe_values = [s["mean_caqe"] for s in dataset_stats.values() if s["mean_caqe"] > 0]

    if not mean_caqe_values:
        raise ValueError("No valid CAQ-E values found across datasets")

    # 1. Cross-dataset variance
    grand_mean = np.mean(mean_caqe_values)
    cross_dataset_variance = np.var(mean_caqe_values)
    mean_caqe_var_percent = (np.std(mean_caqe_values) / grand_mean * 100.0) if grand_mean > 0 else 0.0

    # 2. Energy coherence (correlation between energy and CAQ-E)
    all_energies = []
    all_caqe = []
    for stats_dict in dataset_stats.values():
        all_energies.extend(stats_dict["energies"])
        all_caqe.extend(stats_dict["caqe_values"])

    if len(all_energies) >= 2 and len(all_caqe) >= 2:
        energy_caq_corr, energy_caq_pvalue = stats.pearsonr(all_energies, all_caqe)
    else:
        energy_caq_corr = 0.0
        energy_caq_pvalue = 1.0

    # 3. Drift index (relative to first dataset or baseline)
    baseline_mean = mean_caqe_values[0] if mean_caqe_values else 1.0
    drift_values = [(val - baseline_mean) / baseline_mean * 100.0 for val in mean_caqe_values]
    drift_index = float(np.mean(np.abs(drift_values)))

    # 4. Energy-CPU linearity check
    if len(all_energies) >= 2 and len([ds["cpu_times"] for ds in dataset_stats.values()]):
        all_cpu = []
        for stats_dict in dataset_stats.values():
            all_cpu.extend(stats_dict["cpu_times"])

        if len(all_cpu) == len(all_energies) and len(all_cpu) >= 2:
            energy_cpu_corr, _ = stats.pearsonr(all_cpu, all_energies)
        else:
            energy_cpu_corr = 0.0
    else:
        energy_cpu_corr = 0.0

    stability_metrics = {
        "mean_caqe_var": float(mean_caqe_var_percent),
        "energy_coherence": float(energy_caq_corr),
        "energy_coherence_pvalue": float(energy_caq_pvalue),
        "drift_index": float(drift_index),
        "energy_cpu_correlation": float(energy_cpu_corr),
        "grand_mean_caqe": float(grand_mean),
        "num_datasets": len(dataset_stats),
        "per_dataset_stats": {
            name: {
                "mean_caqe": s["mean_caqe"],
                "baseline_caqe": s["baseline_caqe"],
                "caqe_std": s["caqe_std"],
            }
            for name, s in dataset_stats.items()
        },
        "_raw_data": dataset_stats,  # For plotting
    }

    return stability_metrics


def summarize_audit(stability_metrics: Dict, output_path: Path = None) -> str:
    """
    Generate human-readable audit report.

    Args:
        stability_metrics: Output from compute_stability_metrics()
        output_path: Optional file path to write report

    Returns:
        Report text string
    """
    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append("PHASE H.5.3 - CROSS-DATASET STABILITY AUDIT REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Summary metrics
    report_lines.append("OVERALL STABILITY METRICS")
    report_lines.append("-" * 70)
    report_lines.append(f"Number of Datasets: {stability_metrics['num_datasets']}")
    report_lines.append(f"Grand Mean CAQ-E: {stability_metrics['grand_mean_caqe']:.4f}")
    report_lines.append(f"Cross-Dataset Variance: {stability_metrics['mean_caqe_var']:.2f}%")
    report_lines.append(f"Drift Index: {stability_metrics['drift_index']:.2f}%")
    report_lines.append(f"Energy-CAQ Correlation: {stability_metrics['energy_coherence']:.4f}")
    report_lines.append(f"Energy-CPU Correlation: {stability_metrics['energy_cpu_correlation']:.4f}")
    report_lines.append("")

    # Acceptance criteria checks
    report_lines.append("ACCEPTANCE CRITERIA")
    report_lines.append("-" * 70)

    variance_pass = stability_metrics['mean_caqe_var'] <= 5.0
    drift_pass = stability_metrics['drift_index'] <= 10.0
    coherence_pass = stability_metrics['energy_coherence'] >= 0.8

    report_lines.append(f"✓ Variance ≤ 5%: {'PASS' if variance_pass else 'FAIL'} "
                       f"({stability_metrics['mean_caqe_var']:.2f}%)")
    report_lines.append(f"✓ Drift Index ≤ 10%: {'PASS' if drift_pass else 'FAIL'} "
                       f"({stability_metrics['drift_index']:.2f}%)")
    report_lines.append(f"✓ Energy Coherence ≥ 0.8: {'PASS' if coherence_pass else 'FAIL'} "
                       f"({stability_metrics['energy_coherence']:.4f})")

    all_pass = variance_pass and drift_pass and coherence_pass
    report_lines.append("")
    report_lines.append(f"Overall Status: {'✅ PASS' if all_pass else '⚠ NEEDS REVIEW'}")
    report_lines.append("")

    # Per-dataset breakdown
    report_lines.append("PER-DATASET STATISTICS")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Dataset':<25} {'Mean CAQ-E':<15} {'Baseline':<15} {'Std Dev':<15}")
    report_lines.append("-" * 70)

    for dataset_name, stats in stability_metrics['per_dataset_stats'].items():
        report_lines.append(
            f"{dataset_name:<25} "
            f"{stats['mean_caqe']:<15.4f} "
            f"{stats['baseline_caqe']:<15.4f} "
            f"{stats['caqe_std']:<15.4f}"
        )

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF STABILITY AUDIT REPORT")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    # Write to file if specified
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Audit report written to {output_path}", file=sys.stderr)

    return report_text


def plot_audit_charts(stability_metrics: Dict, output_dir: Path):
    """
    Generate audit visualization charts.

    Args:
        stability_metrics: Output from compute_stability_metrics()
        output_dir: Directory to save PNG files
    """
    if not PLOTTING_AVAILABLE:
        print("⚠ Matplotlib not available. Skipping plot generation.", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = stability_metrics.get("_raw_data", {})

    # 1. CAQ-E vs Energy scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset_name, data in raw_data.items():
        energies = data["energies"]
        caqe_values = data["caqe_values"]

        if len(energies) == len(caqe_values):
            ax.scatter(energies, caqe_values, label=dataset_name, alpha=0.6, s=50)

    ax.set_xlabel("Energy (Joules)")
    ax.set_ylabel("CAQ-E")
    ax.set_title(f"CAQ-E vs Energy (Correlation: {stability_metrics['energy_coherence']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    chart_path = output_dir / "caqe_vs_energy.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {chart_path}", file=sys.stderr)

    # 2. Drift heatmap (per-dataset deviation from mean)
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset_names = list(stability_metrics['per_dataset_stats'].keys())
    mean_caqe_values = [stability_metrics['per_dataset_stats'][name]['mean_caqe']
                        for name in dataset_names]

    grand_mean = stability_metrics['grand_mean_caqe']
    deviations = [(val - grand_mean) / grand_mean * 100.0 for val in mean_caqe_values]

    colors = ['green' if abs(d) <= 5 else 'orange' if abs(d) <= 10 else 'red' for d in deviations]

    bars = ax.barh(dataset_names, deviations, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(-5, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(5, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel("Deviation from Mean (%)")
    ax.set_title("Per-Dataset Drift from Grand Mean CAQ-E")
    ax.grid(True, alpha=0.3, axis='x')

    chart_path = output_dir / "drift_heatmap.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {chart_path}", file=sys.stderr)

    # 3. Variance distribution (box plot)
    fig, ax = plt.subplots(figsize=(10, 6))

    caqe_data = [data["caqe_values"] for data in raw_data.values()]
    labels = list(raw_data.keys())

    bp = ax.boxplot(caqe_data, labels=labels, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_ylabel("CAQ-E")
    ax.set_title("CAQ-E Distribution Across Datasets")
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    chart_path = output_dir / "var_distribution.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {chart_path}", file=sys.stderr)


def compare_results(results1_path: Path, results2_path: Path) -> Dict:
    """
    Compare two benchmark results for reproducibility.

    Args:
        results1_path: Path to first results JSON
        results2_path: Path to second results JSON

    Returns:
        Comparison metrics including drift_index
    """
    with open(results1_path, 'r') as f:
        results1 = json.load(f)

    with open(results2_path, 'r') as f:
        results2 = json.load(f)

    metrics1 = compute_stability_metrics(results1)
    metrics2 = compute_stability_metrics(results2)

    # Compute reproducibility drift
    drift = abs(metrics1['grand_mean_caqe'] - metrics2['grand_mean_caqe']) / metrics1['grand_mean_caqe'] * 100.0

    comparison = {
        "results1_mean_caqe": metrics1['grand_mean_caqe'],
        "results2_mean_caqe": metrics2['grand_mean_caqe'],
        "reproducibility_drift": float(drift),
        "drift_acceptable": drift <= 10.0,
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Dataset Stability Audit (Phase H.5.3)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--results', type=Path, help='Benchmark results JSON file')
    parser.add_argument('--compare', nargs=2, type=Path, metavar=('FILE1', 'FILE2'),
                       help='Compare two results files for reproducibility')
    parser.add_argument('--output-dir', type=Path, default=Path('reports'),
                       help='Output directory for reports and plots (default: reports)')

    args = parser.parse_args()

    if args.compare:
        # Reproducibility comparison mode
        print(f"Comparing {args.compare[0]} vs {args.compare[1]}", file=sys.stderr)
        comparison = compare_results(args.compare[0], args.compare[1])

        print(json.dumps(comparison, indent=2))

        if comparison['drift_acceptable']:
            print(f"✓ Reproducibility drift {comparison['reproducibility_drift']:.2f}% ≤ 10%",
                  file=sys.stderr)
            return 0
        else:
            print(f"⚠ Reproducibility drift {comparison['reproducibility_drift']:.2f}% > 10%",
                  file=sys.stderr)
            return 1

    elif args.results:
        # Single results audit mode
        print(f"Loading results from {args.results}", file=sys.stderr)

        with open(args.results, 'r') as f:
            results = json.load(f)

        # Compute stability metrics
        stability_metrics = compute_stability_metrics(results)

        # Generate report
        report_path = args.output_dir / "phase_h5_3_audit.txt"
        report_text = summarize_audit(stability_metrics, report_path)
        print(report_text)

        # Generate plots
        plots_dir = args.output_dir / "audit_plots"
        plot_audit_charts(stability_metrics, plots_dir)

        print(f"\n✓ Audit complete. Reports in {args.output_dir}", file=sys.stderr)
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

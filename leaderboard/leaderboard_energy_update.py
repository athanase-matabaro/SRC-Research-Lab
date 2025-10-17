#!/usr/bin/env python3
"""
Leaderboard Energy Extension for SRC Research Lab

Extends leaderboard schema and updates with CAQ-E (Energy-Aware) metrics.
Adds energy_joules, caq_e, and device_info fields to submissions.

Phase H.5.1: Adds variance gate filtering to reject unstable runs.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Phase: H.5 - Energy-Aware Compression
Phase: H.5.1 - Runtime Guardrails and Variance Gate
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from energy.runtime_guard import RuntimeGuard


WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()

# Phase H.5.1: Variance gate threshold (IQR/median ≤ 25%)
VARIANCE_GATE_THRESHOLD = 25.0


def load_pass_reports(reports_dir: Path, enforce_variance_gate: bool = True) -> List[Dict]:
    """
    Load all PASS reports from directory.

    Phase H.5.1: Filters out reports with high variance (IQR/median > 25%).

    Args:
        reports_dir: Directory containing validation reports.
        enforce_variance_gate: If True, reject reports with high variance.

    Returns:
        List of valid report entries.
    """
    reports = []
    rejected_count = 0
    guard = RuntimeGuard()

    if not reports_dir.exists():
        return reports

    for report_file in reports_dir.glob("*.json"):
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)

            if report.get("status") == "PASS":
                # Extract key fields
                submission = report.get("submission", {})
                computed = report.get("computed_metrics", report.get("computed", {}))

                # Phase H.5.1: Check variance gate if guardrail info available
                variance_gate_pass = True
                variance_stats = None

                if enforce_variance_gate:
                    # Check if report has guardrail data
                    guardrails = report.get("guardrails")
                    if guardrails:
                        variance_gate_pass = guardrails.get("all_guards_pass", True)
                        variance_stats = guardrails.get("variance_stats")
                    else:
                        # Fallback: check if we have raw run data
                        runs = report.get("runs", [])
                        if runs and len(runs) >= 2:
                            # Extract CAQ-E or CAQ values
                            if "caq_e" in runs[0]:
                                values = [r["caq_e"] for r in runs]
                            elif "caq" in runs[0]:
                                values = [r["caq"] for r in runs]
                            else:
                                values = None

                            if values:
                                variance_gate_pass, variance_stats = guard.check_variance_gate(
                                    values, threshold_percent=VARIANCE_GATE_THRESHOLD
                                )

                # REJECT if variance gate failed
                if not variance_gate_pass:
                    rejected_count += 1
                    variance_pct = variance_stats.get("variance_percent", 0) if variance_stats else 0
                    print(
                        f"⚠ REJECTED (high variance): {report_file.name} "
                        f"(variance: {variance_pct:.1f}% > {VARIANCE_GATE_THRESHOLD}%)",
                        file=sys.stderr
                    )
                    continue  # Skip this report

                entry = {
                    "report_file": report_file.name,
                    "submitter": submission.get("submitter", "unknown"),
                    "dataset": submission.get("dataset", "unknown"),
                    "codec": submission.get("codec", "unknown"),
                    "version": submission.get("version", "unknown"),
                    "computed_caq": computed.get("computed_caq", 0.0),
                    "mean_ratio": computed.get("mean_ratio", 0.0),
                    "mean_cpu": computed.get("mean_cpu", 0.0),
                    "variance": computed.get("variance", 0.0),
                    "timestamp": report.get("timestamp", ""),
                    # Adaptive fields
                    "adaptive_flag": report.get("adaptive_flag", False),
                    "delta_vs_src_baseline": computed.get("delta_vs_src_baseline", 0.0),
                    "validated_by": report.get("validated_by", ""),
                    # Energy fields (Phase H.5)
                    "energy_joules": computed.get("energy_joules"),
                    "caq_e": computed.get("caq_e"),
                    "device_info": computed.get("device_info"),
                    # Guardrail fields (Phase H.5.1)
                    "variance_gate_pass": variance_gate_pass,
                    "variance_stats": variance_stats,
                }
                reports.append(entry)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping invalid report {report_file.name}: {e}", file=sys.stderr)
            continue

    if rejected_count > 0:
        print(f"\n⚠ Rejected {rejected_count} reports due to high variance (>{VARIANCE_GATE_THRESHOLD}%)", file=sys.stderr)

    return reports


def aggregate_by_dataset(reports: List[Dict]) -> Dict[str, List[Dict]]:
    """Aggregate reports by dataset and sort by CAQ-E (if available), else CAQ."""
    datasets = {}

    for report in reports:
        dataset = report["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(report)

    # Sort each dataset by CAQ-E (if available), else CAQ descending
    for dataset in datasets:
        datasets[dataset].sort(
            key=lambda x: (
                x.get("caq_e") if x.get("caq_e") is not None else x["computed_caq"]
            ),
            reverse=True
        )

    return datasets


def compute_stats(entries: List[Dict]) -> Dict:
    """Compute statistics for a list of entries."""
    if not entries:
        return {
            "count": 0,
            "mean_caq": 0.0,
            "median_caq": 0.0,
            "stddev_caq": 0.0,
            "max_caq": 0.0,
            "min_caq": 0.0,
            "mean_caq_e": None,
            "median_caq_e": None,
            "max_caq_e": None,
            "min_caq_e": None,
            "mean_energy_joules": None,
        }

    caqs = [e["computed_caq"] for e in entries]
    caqs_sorted = sorted(caqs)
    n = len(caqs)

    mean_caq = sum(caqs) / n
    median_caq = caqs_sorted[n // 2] if n % 2 == 1 else (caqs_sorted[n//2-1] + caqs_sorted[n//2]) / 2

    # Compute standard deviation
    variance = sum((x - mean_caq) ** 2 for x in caqs) / n
    stddev_caq = variance ** 0.5

    stats = {
        "count": n,
        "mean_caq": round(mean_caq, 2),
        "median_caq": round(median_caq, 2),
        "stddev_caq": round(stddev_caq, 2),
        "max_caq": round(max(caqs), 2),
        "min_caq": round(min(caqs), 2),
    }

    # Compute CAQ-E statistics if available
    caq_es = [e.get("caq_e") for e in entries if e.get("caq_e") is not None]
    if caq_es:
        caq_es_sorted = sorted(caq_es)
        n_e = len(caq_es)
        mean_caq_e = sum(caq_es) / n_e
        median_caq_e = caq_es_sorted[n_e // 2] if n_e % 2 == 1 else (caq_es_sorted[n_e//2-1] + caq_es_sorted[n_e//2]) / 2

        stats.update({
            "mean_caq_e": round(mean_caq_e, 2),
            "median_caq_e": round(median_caq_e, 2),
            "max_caq_e": round(max(caq_es), 2),
            "min_caq_e": round(min(caq_es), 2),
        })
    else:
        stats.update({
            "mean_caq_e": None,
            "median_caq_e": None,
            "max_caq_e": None,
            "min_caq_e": None,
        })

    # Compute energy statistics if available
    energies = [e.get("energy_joules") for e in entries if e.get("energy_joules") is not None]
    if energies:
        mean_energy = sum(energies) / len(energies)
        stats["mean_energy_joules"] = round(mean_energy, 6)
    else:
        stats["mean_energy_joules"] = None

    return stats


def generate_json(datasets: Dict[str, List[Dict]], output_path: Path):
    """Generate leaderboard.json with energy fields."""
    leaderboard = {
        "generated_at": datetime.now().isoformat(),
        "version": "2.0",  # Version 2.0 includes energy metrics
        "datasets": {}
    }

    all_entries = []

    for dataset_name, entries in datasets.items():
        leaderboard["datasets"][dataset_name] = {
            "entries": entries,
            "stats": compute_stats(entries)
        }
        all_entries.extend(entries)

    # Global ranking by CAQ-E (if available), else CAQ
    all_entries.sort(
        key=lambda x: (
            x.get("caq_e") if x.get("caq_e") is not None else x["computed_caq"]
        ),
        reverse=True
    )
    leaderboard["global"] = {
        "top": all_entries[:10],
        "stats": compute_stats(all_entries)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    print(f"✓ Generated {output_path}")


def generate_markdown(datasets: Dict[str, List[Dict]], output_path: Path):
    """Generate leaderboard.md with CAQ-E column."""
    lines = []
    lines.append("# SRC Research Lab — CAQ Leaderboard\n\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    lines.append("## Overview\n\n")
    lines.append("This leaderboard ranks compression algorithms by their CAQ (Compression-Accuracy Quotient) score.\n")
    lines.append("CAQ balances compression ratio with computational efficiency:\n\n")
    lines.append("```\nCAQ = compression_ratio / (cpu_seconds + 1)\n```\n\n")

    lines.append("**NEW: CAQ-E (Energy-Aware)** balances compression with both time and energy:\n\n")
    lines.append("```\nCAQ-E = compression_ratio / (energy_joules + cpu_seconds)\n```\n\n")

    # Collect all adaptive entries for Adaptive Top 5
    all_entries = []
    for dataset_entries in datasets.values():
        all_entries.extend(dataset_entries)

    adaptive_entries = [e for e in all_entries if e.get("adaptive_flag", False)]

    if adaptive_entries:
        # Sort by CAQ-E if available, else CAQ
        adaptive_entries.sort(
            key=lambda x: (
                x.get("caq_e") if x.get("caq_e") is not None else x["computed_caq"]
            ),
            reverse=True
        )
        lines.append("## 🔬 Adaptive Top 5\n\n")
        lines.append("*Adaptive Learned Compression Model (ALCM) results with neural entropy modeling*\n\n")

        # Check if any entry has CAQ-E
        has_caq_e = any(e.get("caq_e") is not None for e in adaptive_entries[:5])

        if has_caq_e:
            lines.append("| Rank | Submitter | Dataset | CAQ | CAQ-E | Energy (J) | Δ vs Baseline | Variance (%) |\n")
            lines.append("|------|-----------|---------|-----|-------|------------|---------------|-------------|\n")
        else:
            lines.append("| Rank | Submitter | Dataset | CAQ | Δ vs Baseline | Ratio | Variance (%) |\n")
            lines.append("|------|-----------|---------|-----|---------------|-------|-------------|\n")

        for rank, entry in enumerate(adaptive_entries[:5], 1):
            delta = entry.get("delta_vs_src_baseline", 0.0)
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"

            if has_caq_e:
                caq_e = entry.get("caq_e")
                caq_e_str = f"{caq_e:.2f}" if caq_e is not None else "N/A"
                energy = entry.get("energy_joules")
                energy_str = f"{energy:.4f}" if energy is not None else "N/A"

                lines.append(f"| {rank} | {entry['submitter']} | {entry['dataset']} | ")
                lines.append(f"{entry['computed_caq']:.2f} | {caq_e_str} | {energy_str} | ")
                lines.append(f"{delta_str} | {entry['variance']:.2f} |\n")
            else:
                lines.append(f"| {rank} | {entry['submitter']} | {entry['dataset']} | ")
                lines.append(f"{entry['computed_caq']:.2f} | {delta_str} | ")
                lines.append(f"{entry['mean_ratio']:.2f} | {entry['variance']:.2f} |\n")

        lines.append("\n")

    for dataset_name in ["text_medium", "image_small", "mixed_stream", "synthetic_gradients"]:
        if dataset_name not in datasets or not datasets[dataset_name]:
            continue

        entries = datasets[dataset_name]
        stats = compute_stats(entries)

        lines.append(f"## Dataset: {dataset_name}\n\n")
        lines.append(f"**Submissions:** {stats['count']} | ")
        lines.append(f"**Mean CAQ:** {stats['mean_caq']:.2f} | ")
        lines.append(f"**Median CAQ:** {stats['median_caq']:.2f}")

        if stats.get("mean_caq_e") is not None:
            lines.append(f" | **Mean CAQ-E:** {stats['mean_caq_e']:.2f}")
            if stats.get("mean_energy_joules") is not None:
                lines.append(f" | **Mean Energy:** {stats['mean_energy_joules']:.4f}J")

        lines.append("\n\n")

        # Check if any entry has CAQ-E
        has_caq_e = any(e.get("caq_e") is not None for e in entries[:10])

        if has_caq_e:
            lines.append("| Rank | Submitter | Codec | CAQ | CAQ-E | Energy (J) | CPU (s) | Variance (%) |\n")
            lines.append("|------|-----------|-------|-----|-------|------------|---------|-------------|\n")
        else:
            lines.append("| Rank | Submitter | Codec | CAQ | Ratio | CPU (s) | Variance (%) |\n")
            lines.append("|------|-----------|-------|-----|-------|---------|-------------|\n")

        for rank, entry in enumerate(entries[:10], 1):
            adaptive_marker = " 🔬" if entry.get("adaptive_flag", False) else ""

            if has_caq_e:
                caq_e = entry.get("caq_e")
                caq_e_str = f"{caq_e:.2f}" if caq_e is not None else "N/A"
                energy = entry.get("energy_joules")
                energy_str = f"{energy:.4f}" if energy is not None else "N/A"

                lines.append(f"| {rank} | {entry['submitter']}{adaptive_marker} | {entry['codec']} | ")
                lines.append(f"{entry['computed_caq']:.2f} | {caq_e_str} | {energy_str} | ")
                lines.append(f"{entry['mean_cpu']:.3f} | {entry['variance']:.2f} |\n")
            else:
                lines.append(f"| {rank} | {entry['submitter']}{adaptive_marker} | {entry['codec']} | ")
                lines.append(f"{entry['computed_caq']:.2f} | {entry['mean_ratio']:.2f} | ")
                lines.append(f"{entry['mean_cpu']:.3f} | {entry['variance']:.2f} |\n")

        lines.append("\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"✓ Generated {output_path}")


def update_schema(schema_path: Path):
    """Update leaderboard schema to include energy fields."""
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Add energy fields to properties
    schema["properties"]["energy_joules"] = {
        "type": "number",
        "minimum": 0,
        "description": "Mean energy consumption in joules across runs (Phase H.5)"
    }

    schema["properties"]["caq_e"] = {
        "type": "number",
        "exclusiveMinimum": 0,
        "description": "Compression-Accuracy-Energy Quotient (CAQ-E = ratio / (joules + seconds))"
    }

    schema["properties"]["device_info"] = {
        "type": "object",
        "description": "Device information for energy measurements",
        "properties": {
            "cpu_model": {
                "type": "string",
                "description": "CPU model name"
            },
            "cores": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of physical cores"
            },
            "threads": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of threads"
            },
            "base_freq_mhz": {
                "type": "number",
                "minimum": 0,
                "description": "Base frequency in MHz"
            }
        }
    }

    # Update runs items to include energy
    schema["properties"]["runs"]["items"]["properties"]["energy_joules"] = {
        "type": "number",
        "minimum": 0,
        "description": "Energy consumed in joules for this run"
    }

    # Update schema description
    schema["description"] = "Schema for submitting CAQ and CAQ-E benchmark results to the SRC Research Lab leaderboard (v2.0 - Energy-Aware)"

    # Save updated schema
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f"✓ Updated schema: {schema_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate energy-aware leaderboard from validated reports (Phase H.5.1: with variance gate)'
    )
    parser.add_argument(
        '--reports-dir',
        required=True,
        help='Directory containing validation reports'
    )
    parser.add_argument(
        '--out-json',
        required=True,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--out-md',
        required=True,
        help='Output markdown file path'
    )
    parser.add_argument(
        '--schema',
        help='Path to schema file to update (optional)'
    )
    parser.add_argument(
        '--no-variance-gate',
        action='store_true',
        help='Disable variance gate filtering (Phase H.5.1)'
    )

    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)

    # Update schema if provided
    if args.schema:
        schema_path = Path(args.schema)
        if schema_path.exists():
            update_schema(schema_path)
        else:
            print(f"Warning: Schema file not found: {schema_path}", file=sys.stderr)

    # Load PASS reports with variance gate (Phase H.5.1)
    enforce_variance_gate = not args.no_variance_gate
    if enforce_variance_gate:
        print(f"Phase H.5.1: Variance gate enabled (threshold: {VARIANCE_GATE_THRESHOLD}%)")
    else:
        print("Variance gate disabled")

    reports = load_pass_reports(reports_dir, enforce_variance_gate=enforce_variance_gate)

    if not reports:
        print("ERROR: No PASS reports found", file=sys.stderr)
        return 5

    print(f"Loaded {len(reports)} PASS reports")

    # Count reports with energy data
    energy_reports = [r for r in reports if r.get("energy_joules") is not None]
    if energy_reports:
        print(f"  {len(energy_reports)} reports include energy measurements")

    # Aggregate by dataset
    datasets = aggregate_by_dataset(reports)

    # Generate outputs
    generate_json(datasets, Path(args.out_json))
    generate_markdown(datasets, Path(args.out_md))

    print(f"\n✓ Energy-aware leaderboard updated successfully")
    print(f"  Datasets: {', '.join(datasets.keys())}")
    print(f"  Total submissions: {len(reports)}")
    print(f"  Energy-enabled: {len(energy_reports)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

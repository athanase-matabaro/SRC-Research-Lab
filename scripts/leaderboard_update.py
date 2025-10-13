#!/usr/bin/env python3
"""
Leaderboard Aggregator for SRC Research Lab

Generates leaderboard.json and leaderboard.md from validated reports.
Exit codes:
    0: Success
    5: No PASS reports found
    6: Verification failure
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()


def load_pass_reports(reports_dir: Path) -> list:
    """Load all PASS reports from directory."""
    reports = []
    
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
                    "timestamp": report.get("timestamp", "")
                }
                reports.append(entry)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping invalid report {report_file.name}: {e}", file=sys.stderr)
            continue
    
    return reports


def aggregate_by_dataset(reports: list) -> dict:
    """Aggregate reports by dataset and sort by CAQ."""
    datasets = {}
    
    for report in reports:
        dataset = report["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(report)
    
    # Sort each dataset by CAQ descending
    for dataset in datasets:
        datasets[dataset].sort(key=lambda x: x["computed_caq"], reverse=True)
    
    return datasets


def compute_stats(entries: list) -> dict:
    """Compute statistics for a list of entries."""
    if not entries:
        return {
            "count": 0,
            "mean_caq": 0.0,
            "median_caq": 0.0,
            "stddev_caq": 0.0,
            "max_caq": 0.0,
            "min_caq": 0.0
        }
    
    caqs = [e["computed_caq"] for e in entries]
    caqs_sorted = sorted(caqs)
    n = len(caqs)
    
    mean_caq = sum(caqs) / n
    median_caq = caqs_sorted[n // 2] if n % 2 == 1 else (caqs_sorted[n//2-1] + caqs_sorted[n//2]) / 2
    
    # Compute standard deviation
    variance = sum((x - mean_caq) ** 2 for x in caqs) / n
    stddev_caq = variance ** 0.5
    
    return {
        "count": n,
        "mean_caq": round(mean_caq, 2),
        "median_caq": round(median_caq, 2),
        "stddev_caq": round(stddev_caq, 2),
        "max_caq": round(max(caqs), 2),
        "min_caq": round(min(caqs), 2)
    }


def generate_json(datasets: dict, output_path: Path):
    """Generate leaderboard.json."""
    leaderboard = {
        "generated_at": datetime.now().isoformat(),
        "datasets": {}
    }
    
    all_entries = []
    
    for dataset_name, entries in datasets.items():
        leaderboard["datasets"][dataset_name] = {
            "entries": entries,
            "stats": compute_stats(entries)
        }
        all_entries.extend(entries)
    
    # Global ranking
    all_entries.sort(key=lambda x: x["computed_caq"], reverse=True)
    leaderboard["global"] = {
        "top": all_entries[:10],
        "stats": compute_stats(all_entries)
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    print(f"✓ Generated {output_path}")


def generate_markdown(datasets: dict, output_path: Path):
    """Generate leaderboard.md."""
    lines = []
    lines.append("# SRC Research Lab — CAQ Leaderboard\n\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    
    lines.append("## Overview\n\n")
    lines.append("This leaderboard ranks compression algorithms by their CAQ (Compression-Accuracy Quotient) score.\n")
    lines.append("CAQ balances compression ratio with computational efficiency:\n\n")
    lines.append("```\nCAQ = compression_ratio / (cpu_seconds + 1)\n```\n\n")
    
    for dataset_name in ["text_medium", "image_small", "mixed_stream"]:
        if dataset_name not in datasets or not datasets[dataset_name]:
            continue
        
        entries = datasets[dataset_name]
        stats = compute_stats(entries)
        
        lines.append(f"## Dataset: {dataset_name}\n\n")
        lines.append(f"**Submissions:** {stats['count']} | ")
        lines.append(f"**Mean CAQ:** {stats['mean_caq']:.2f} | ")
        lines.append(f"**Median CAQ:** {stats['median_caq']:.2f}\n\n")
        
        lines.append("| Rank | Submitter | Codec | CAQ | Ratio | CPU (s) | Variance (%) |\n")
        lines.append("|------|-----------|-------|-----|-------|---------|-------------|\n")
        
        for rank, entry in enumerate(entries[:10], 1):
            lines.append(f"| {rank} | {entry['submitter']} | {entry['codec']} | ")
            lines.append(f"{entry['computed_caq']:.2f} | {entry['mean_ratio']:.2f} | ")
            lines.append(f"{entry['mean_cpu']:.3f} | {entry['variance']:.2f} |\n")
        
        lines.append("\n")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(lines)
    
    print(f"✓ Generated {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate leaderboard from validated reports')
    parser.add_argument('--reports-dir', required=True, help='Directory containing validation reports')
    parser.add_argument('--out-json', required=True, help='Output JSON file path')
    parser.add_argument('--out-md', required=True, help='Output markdown file path')
    parser.add_argument('--verify-top', type=int, default=0, help='Verify top N submissions (not implemented)')
    
    args = parser.parse_args()
    
    reports_dir = Path(args.reports_dir)
    
    # Load PASS reports
    reports = load_pass_reports(reports_dir)
    
    if not reports:
        print("ERROR: No PASS reports found", file=sys.stderr)
        return 5
    
    print(f"Loaded {len(reports)} PASS reports")
    
    # Aggregate by dataset
    datasets = aggregate_by_dataset(reports)
    
    # Generate outputs
    generate_json(datasets, Path(args.out_json))
    generate_markdown(datasets, Path(args.out_md))
    
    print(f"\n✓ Leaderboard updated successfully")
    print(f"  Datasets: {', '.join(datasets.keys())}")
    print(f"  Total submissions: {len(reports)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

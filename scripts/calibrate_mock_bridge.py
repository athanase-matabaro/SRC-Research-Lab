#!/usr/bin/env python3
"""
Mock Bridge Calibration Tool (Phase H.5.2)

Computes empirical statistical moments from real private benchmark runs
and fits parametric distributions to create calibration files for improved
mock bridge fidelity.

This tool operates on PRIVATE data and should ONLY be run locally.
Calibration files must NOT be committed to version control.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Phase: H.5.2 - Mock-Bridge Fidelity Calibration
"""

import sys
import json
import argparse
import glob
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats


class CalibrationTool:
    """Fit parametric distributions to empirical benchmark data."""

    def __init__(self, dataset_name: str, fit_distributions: List[str] = None):
        """
        Initialize calibration tool.

        Args:
            dataset_name: Name of dataset being calibrated
            fit_distributions: List of distributions to fit (default: ['lognormal', 'gamma'])
        """
        self.dataset_name = dataset_name
        self.fit_distributions = fit_distributions or ['lognormal', 'gamma']

    def load_benchmark_runs(self, input_dir: Path) -> List[Dict]:
        """
        Load benchmark run results from directory.

        Args:
            input_dir: Directory containing JSON result files

        Returns:
            List of run result dictionaries
        """
        runs = []

        # Search for JSON files
        json_files = list(input_dir.glob("*.json"))

        if not json_files:
            raise ValueError(f"No JSON files found in {input_dir}")

        print(f"Found {len(json_files)} JSON files", file=sys.stderr)

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Handle different formats (single run, list of runs, nested structure)
                if isinstance(data, list):
                    runs.extend(data)
                elif isinstance(data, dict):
                    # Check for runs array
                    if "runs" in data:
                        runs.extend(data["runs"])
                    else:
                        runs.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping invalid file {json_file.name}: {e}", file=sys.stderr)
                continue

        print(f"Loaded {len(runs)} total runs", file=sys.stderr)
        return runs

    def extract_metrics(self, runs: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Extract metrics arrays from runs.

        Args:
            runs: List of run dictionaries

        Returns:
            Dictionary mapping metric names to numpy arrays
        """
        metrics = {
            "compression_ratio": [],
            "cpu_seconds": [],
            "energy_joules": [],
        }

        for run in runs:
            # Handle different key names
            ratio = run.get("compression_ratio") or run.get("ratio")
            cpu = run.get("cpu_seconds") or run.get("cpu_time")
            energy = run.get("energy_joules") or run.get("joules")

            if ratio is not None and np.isfinite(ratio):
                metrics["compression_ratio"].append(ratio)
            if cpu is not None and np.isfinite(cpu):
                metrics["cpu_seconds"].append(cpu)
            if energy is not None and np.isfinite(energy):
                metrics["energy_joules"].append(energy)

        # Convert to numpy arrays
        for key in metrics:
            metrics[key] = np.array(metrics[key])

        # Validate
        for key, values in metrics.items():
            if len(values) == 0:
                warnings.warn(f"No valid values for {key}")

        return metrics

    def compute_empirical_stats(self, values: np.ndarray) -> Dict:
        """Compute empirical statistical moments."""
        if len(values) == 0:
            return {}

        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "variance": float(np.var(values)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "q95": float(np.percentile(values, 95)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "skewness": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values)),
        }

    def fit_lognormal(self, values: np.ndarray) -> Tuple[Dict, float]:
        """
        Fit log-normal distribution to data.

        Returns:
            (params, ks_pvalue): Distribution parameters and goodness-of-fit p-value
        """
        if len(values) < 3:
            return {}, 0.0

        # Fit log-normal
        shape, loc, scale = stats.lognorm.fit(values, floc=0)

        # Compute KS test
        ks_stat, ks_pvalue = stats.kstest(values, lambda x: stats.lognorm.cdf(x, shape, loc, scale))

        # Convert to mu, sigma parameterization
        mu = np.log(scale)
        sigma = shape

        params = {
            "distribution": "lognormal",
            "params": {
                "mu": float(mu),
                "sigma": float(sigma)
            },
            "scipy_params": {
                "shape": float(shape),
                "loc": float(loc),
                "scale": float(scale)
            }
        }

        return params, float(ks_pvalue)

    def fit_gamma(self, values: np.ndarray) -> Tuple[Dict, float]:
        """
        Fit gamma distribution to data.

        Returns:
            (params, ks_pvalue): Distribution parameters and goodness-of-fit p-value
        """
        if len(values) < 3:
            return {}, 0.0

        # Fit gamma
        shape, loc, scale = stats.gamma.fit(values, floc=0)

        # Compute KS test
        ks_stat, ks_pvalue = stats.kstest(values, lambda x: stats.gamma.cdf(x, shape, loc, scale))

        params = {
            "distribution": "gamma",
            "params": {
                "shape": float(shape),
                "scale": float(scale)
            },
            "scipy_params": {
                "shape": float(shape),
                "loc": float(loc),
                "scale": float(scale)
            }
        }

        return params, float(ks_pvalue)

    def fit_best_distribution(self, values: np.ndarray, metric_name: str) -> Dict:
        """
        Fit multiple distributions and select best based on KS test.

        Args:
            values: Data to fit
            metric_name: Name of metric for logging

        Returns:
            Best fit parameters with diagnostics
        """
        if len(values) < 3:
            print(f"Warning: Not enough data for {metric_name} ({len(values)} samples)", file=sys.stderr)
            return {}

        fits = {}

        # Try each distribution
        if 'lognormal' in self.fit_distributions:
            try:
                params, ks_pvalue = self.fit_lognormal(values)
                if params:
                    fits['lognormal'] = (params, ks_pvalue)
                    print(f"  {metric_name} log-normal KS p-value: {ks_pvalue:.4f}", file=sys.stderr)
            except Exception as e:
                print(f"  Warning: Failed to fit lognormal to {metric_name}: {e}", file=sys.stderr)

        if 'gamma' in self.fit_distributions:
            try:
                params, ks_pvalue = self.fit_gamma(values)
                if params:
                    fits['gamma'] = (params, ks_pvalue)
                    print(f"  {metric_name} gamma KS p-value: {ks_pvalue:.4f}", file=sys.stderr)
            except Exception as e:
                print(f"  Warning: Failed to fit gamma to {metric_name}: {e}", file=sys.stderr)

        if not fits:
            return {}

        # Select best fit (highest KS p-value)
        best_name, (best_params, best_pvalue) = max(fits.items(), key=lambda x: x[1][1])

        print(f"  → Best fit for {metric_name}: {best_name} (p={best_pvalue:.4f})", file=sys.stderr)

        # Add diagnostics
        best_params["ks_pvalue"] = best_pvalue
        best_params["ks_acceptable"] = best_pvalue >= 0.01

        return best_params

    def calibrate(self, runs: List[Dict]) -> Dict:
        """
        Perform full calibration.

        Args:
            runs: List of benchmark run results

        Returns:
            Calibration dictionary
        """
        print(f"\nCalibrating {self.dataset_name}...", file=sys.stderr)

        # Extract metrics
        metrics = self.extract_metrics(runs)

        calibration = {
            "dataset": self.dataset_name,
            "calibration_date": "2025-10-17",
            "num_runs": len(runs),
            "note": "PRIVATE calibration file. DO NOT commit to version control.",
        }

        # Fit distributions for each metric
        for metric_name, values in metrics.items():
            if len(values) == 0:
                print(f"Skipping {metric_name} (no data)", file=sys.stderr)
                continue

            print(f"\nFitting {metric_name} ({len(values)} samples):", file=sys.stderr)

            # Compute empirical stats
            empirical = self.compute_empirical_stats(values)

            # Fit distribution
            fit_result = self.fit_best_distribution(values, metric_name)

            # Store results
            calibration[metric_name] = fit_result if fit_result else {}
            calibration[f"{metric_name}_empirical"] = empirical

        # Add derived metrics
        calibration["compressed_size_bytes"] = {"method": "derived"}

        # Estimate noise model from empirical variance
        if "cpu_seconds" in metrics and len(metrics["cpu_seconds"]) > 0:
            cpu_std = np.std(metrics["cpu_seconds"])
            cpu_mean = np.mean(metrics["cpu_seconds"])
            jitter_sigma = min(0.003, cpu_std * 0.1)  # 10% of std, capped

            calibration["noise_model"] = {
                "cpu_jitter_sigma": float(jitter_sigma),
                "ratio_multiplier_sigma": 0.02,
                "tail_mixture_weight": 0.05,
                "tail_scale_factor": 1.5
            }

        return calibration


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate mock bridge from private benchmark runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: This tool operates on PRIVATE data.
Calibration files must NOT be committed to version control.
Use .gitignore to exclude release/mock_bridge_calibration/*.json

Example:
  python scripts/calibrate_mock_bridge.py \\
    --input-dir results/private_runs/text_medium \\
    --out-file release/mock_bridge_calibration/text_medium.json \\
    --fit lognormal,gamma
        """
    )

    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Directory containing private benchmark JSON results')
    parser.add_argument('--out-file', type=Path, required=True,
                        help='Output calibration JSON file')
    parser.add_argument('--fit', type=str, default='lognormal,gamma',
                        help='Comma-separated list of distributions to fit (default: lognormal,gamma)')
    parser.add_argument('--format', type=str, default='json', choices=['json'],
                        help='Output format (default: json)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing calibration file without prompting')

    args = parser.parse_args()

    # Parse dataset name from output file
    dataset_name = args.out_file.stem

    # Safety check: warn if output is in tracked directory
    if not args.force and args.out_file.exists():
        response = input(f"File {args.out_file} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.", file=sys.stderr)
            return 1

    # Check if output path looks like it might be tracked
    if "public" in str(args.out_file) or "release" in str(args.out_file).split("/")[:2]:
        print("\n⚠️  WARNING: Output path may be in a tracked directory!", file=sys.stderr)
        print("   Calibration files contain PRIVATE data and must NOT be committed.", file=sys.stderr)
        print("   Ensure release/mock_bridge_calibration/* is in .gitignore\n", file=sys.stderr)

        if not args.force:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.", file=sys.stderr)
                return 1

    # Initialize calibration tool
    fit_distributions = args.fit.split(',')
    calibrator = CalibrationTool(dataset_name, fit_distributions)

    # Load runs
    print(f"Loading benchmark runs from {args.input_dir}", file=sys.stderr)
    runs = calibrator.load_benchmark_runs(args.input_dir)

    if not runs:
        print("ERROR: No runs loaded", file=sys.stderr)
        return 1

    # Perform calibration
    calibration = calibrator.calibrate(runs)

    # Write output
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\n✓ Calibration written to {args.out_file}", file=sys.stderr)
    print(f"  Dataset: {calibration['dataset']}", file=sys.stderr)
    print(f"  Runs: {calibration['num_runs']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

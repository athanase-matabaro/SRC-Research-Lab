"""
Energy-Aware Compression Benchmark Runner

Runs comprehensive energy benchmarks on gradient datasets, measuring CAQ-E metrics.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-16
Phase: H.5 - Energy-Aware Compression
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from energy.profiler import EnergyProfiler
from energy.datasets.generate_gradients import load_gradient_dataset
from metrics.caq_energy_metric import (
    compute_caq_and_caqe,
    compute_caqe_delta,
    validate_caqe_threshold,
    compute_energy_variance,
    PHASE_H5_CAQE_THRESHOLD,
)


def compress_gradient_baseline(gradient_data: np.ndarray) -> bytes:
    """
    Baseline compression using NumPy savez_compressed.

    Args:
        gradient_data: Gradient tensor to compress.

    Returns:
        Compressed bytes.
    """
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=gradient_data)
    return buffer.getvalue()


def compress_gradient_adaptive(gradient_data: np.ndarray) -> bytes:
    """
    Adaptive compression with quantization and pruning.

    Simulates the adaptive model from Phase H.3/H.4.

    Args:
        gradient_data: Gradient tensor to compress.

    Returns:
        Compressed bytes.
    """
    import io

    # Adaptive quantization (reduce precision based on entropy)
    mean_abs = np.mean(np.abs(gradient_data))
    scale = max(mean_abs * 0.1, 1e-6)

    quantized = np.round(gradient_data / scale) * scale

    # Adaptive pruning (remove small values)
    threshold = np.percentile(np.abs(quantized), 20)  # Prune bottom 20%
    mask = np.abs(quantized) > threshold
    pruned = quantized * mask

    # Compress the result
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=pruned.astype(np.float16))  # Also use float16
    return buffer.getvalue()


def run_single_benchmark(
    gradient_data: np.ndarray,
    method: str = "baseline",
    num_runs: int = 3
) -> Dict:
    """
    Run benchmark on single gradient sample.

    Args:
        gradient_data: Gradient tensor to compress.
        method: Compression method ("baseline" or "adaptive").
        num_runs: Number of runs for averaging.

    Returns:
        Dictionary with benchmark results.
    """
    compress_func = (
        compress_gradient_baseline if method == "baseline"
        else compress_gradient_adaptive
    )

    original_size = gradient_data.nbytes
    results = {
        "method": method,
        "runs": [],
    }

    for run_idx in range(num_runs):
        with EnergyProfiler() as profiler:
            compressed_data = compress_func(gradient_data)

        joules, seconds = profiler.read()
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size

        metrics = compute_caq_and_caqe(compression_ratio, seconds, joules)

        run_result = {
            "run": run_idx,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "cpu_seconds": seconds,
            "energy_joules": joules,
            **metrics,
        }

        results["runs"].append(run_result)

    # Compute averages
    avg_ratio = np.mean([r["compression_ratio"] for r in results["runs"]])
    avg_caq = np.mean([r["caq"] for r in results["runs"]])
    avg_caqe = np.mean([r["caq_e"] for r in results["runs"]])
    avg_joules = np.mean([r["energy_joules"] for r in results["runs"]])
    avg_seconds = np.mean([r["cpu_seconds"] for r in results["runs"]])
    avg_power = np.mean([r["avg_power_watts"] for r in results["runs"]])

    # Compute variance
    energy_variance = compute_energy_variance(
        [r["energy_joules"] for r in results["runs"]],
        relative=True
    )

    results["averages"] = {
        "compression_ratio": avg_ratio,
        "caq": avg_caq,
        "caq_e": avg_caqe,
        "energy_joules": avg_joules,
        "cpu_seconds": avg_seconds,
        "avg_power_watts": avg_power,
        "energy_variance_percent": energy_variance,
    }

    return results


def run_energy_benchmark_suite(
    datasets: Dict[str, Path],
    output_path: Path,
    num_runs: int = 3
) -> Dict:
    """
    Run complete energy benchmark suite.

    Args:
        datasets: Dictionary mapping dataset name to file path.
        output_path: Output path for results JSON.
        num_runs: Number of runs per benchmark.

    Returns:
        Complete benchmark results.
    """
    print("=" * 70)
    print("PHASE H.5 - ENERGY-AWARE COMPRESSION BENCHMARK")
    print("=" * 70)

    cpu_info = EnergyProfiler.get_cpu_info()
    print(f"\nCPU: {cpu_info['model']}")
    print(f"Cores: {cpu_info['cores']} | Threads: {cpu_info['threads']}")

    # Detect energy measurement method
    profiler = EnergyProfiler()
    print(f"Energy Measurement: {profiler.method.upper()}")
    if profiler.method == "constant":
        print(f"  Using constant power model: {profiler.constant_power}W")
    print()

    all_results = {
        "metadata": {
            "phase": "H.5",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cpu_info": cpu_info,
            "energy_method": profiler.method,
            "num_runs_per_test": num_runs,
        },
        "datasets": {},
    }

    for dataset_name, dataset_path in datasets.items():
        print(f"Processing dataset: {dataset_name}")
        print(f"  Loading from: {dataset_path}")

        # Load dataset
        if dataset_path.suffix == '.npy':
            # Simple numpy array
            data = np.load(dataset_path)
            # Use first sample
            gradient = data[0] if len(data.shape) > 2 else data
        else:
            # Pickled dictionary (CIFAR-10 gradients)
            data = load_gradient_dataset(dataset_path)
            # Extract first epoch, first layer
            first_epoch = list(data.values())[0]
            gradient = list(first_epoch.values())[0]

        print(f"  Gradient shape: {gradient.shape}")
        print(f"  Size: {gradient.nbytes / 1024:.2f} KB")

        # Run baseline
        print(f"  Running baseline compression...")
        baseline_results = run_single_benchmark(gradient, "baseline", num_runs)

        # Run adaptive
        print(f"  Running adaptive compression...")
        adaptive_results = run_single_benchmark(gradient, "adaptive", num_runs)

        # Compute comparison
        baseline_caqe = baseline_results["averages"]["caq_e"]
        adaptive_caqe = adaptive_results["averages"]["caq_e"]
        delta_caqe = compute_caqe_delta(adaptive_caqe, baseline_caqe)
        threshold_met = validate_caqe_threshold(
            adaptive_caqe, baseline_caqe, PHASE_H5_CAQE_THRESHOLD
        )

        dataset_result = {
            "dataset_path": str(dataset_path),
            "gradient_shape": list(gradient.shape),
            "gradient_size_bytes": int(gradient.nbytes),
            "baseline": baseline_results,
            "adaptive": adaptive_results,
            "comparison": {
                "delta_caqe_percent": float(delta_caqe),
                "threshold_percent": float(PHASE_H5_CAQE_THRESHOLD),
                "threshold_met": bool(threshold_met),
            },
        }

        all_results["datasets"][dataset_name] = dataset_result

        # Print summary
        print(f"  Results:")
        print(f"    Baseline CAQ-E:  {baseline_caqe:.6f}")
        print(f"    Adaptive CAQ-E:  {adaptive_caqe:.6f}")
        print(f"    Delta:           {delta_caqe:+.2f}%")
        print(f"    Threshold Met:   {'✓ YES' if threshold_met else '✗ NO'}")
        print()

    # Compute overall statistics
    all_deltas = [
        r["comparison"]["delta_caqe_percent"]
        for r in all_results["datasets"].values()
    ]
    all_thresholds_met = [
        r["comparison"]["threshold_met"]
        for r in all_results["datasets"].values()
    ]

    all_results["summary"] = {
        "num_datasets": len(datasets),
        "mean_delta_caqe_percent": float(np.mean(all_deltas)),
        "min_delta_caqe_percent": float(np.min(all_deltas)),
        "max_delta_caqe_percent": float(np.max(all_deltas)),
        "num_threshold_met": int(sum(all_thresholds_met)),
        "all_thresholds_met": bool(all(all_thresholds_met)),
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Datasets Tested: {all_results['summary']['num_datasets']}")
    print(f"Mean CAQ-E Improvement: {all_results['summary']['mean_delta_caqe_percent']:.2f}%")
    print(f"Range: {all_results['summary']['min_delta_caqe_percent']:.2f}% to "
          f"{all_results['summary']['max_delta_caqe_percent']:.2f}%")
    print(f"Threshold Met: {all_results['summary']['num_threshold_met']}/{len(datasets)}")
    print(f"Overall Status: {'✓ PASS' if all_results['summary']['all_thresholds_met'] else '✗ FAIL'}")
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Phase H.5 energy-aware compression benchmarks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_H5_results.json"),
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per benchmark (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    import random
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Define datasets to test
    base_dir = Path(__file__).parent.parent / "energy" / "datasets"

    datasets = {
        "synthetic_gradients": base_dir / "synthetic_gradients.npy",
        "cifar10_resnet8": base_dir / "real_gradients_cifar10.pkl",
        "mixed_gradients": base_dir / "mixed_gradients.pkl",
    }

    # Verify datasets exist
    missing = [name for name, path in datasets.items() if not path.exists()]
    if missing:
        print(f"ERROR: Missing datasets: {', '.join(missing)}")
        print("\nRun this first:")
        print("  python3 energy/datasets/generate_gradients.py")
        sys.exit(1)

    # Run benchmarks
    run_energy_benchmark_suite(datasets, args.output, args.runs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Mock Bridge - Statistical Compression Emulator (Phase H.5.2)

Provides parametric statistical emulation of SRC engine compression outputs
for external reproducibility without exposing proprietary algorithms.

Features:
- Parametric distribution-based emulation (log-normal, gamma)
- Deterministic seeded sampling for reproducibility
- Calibration file support for improved fidelity
- Replay mode for exact trace reproduction
- Noise models for measurement variance (CPU jitter, energy noise)

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Phase: H.5.2 - Mock-Bridge Fidelity Calibration
"""

import sys
import json
import argparse
import hashlib
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


class MockBridgeEmulator:
    """
    Statistical emulator for compression benchmarks.

    Emulates the statistical properties (mean, variance, tail behavior) of
    SRC engine outputs without replicating the actual compression algorithm.
    """

    def __init__(self, dataset: str, seed: int = 42, calibration_file: Optional[Path] = None):
        """
        Initialize emulator.

        Args:
            dataset: Dataset name (text_medium, image_small, mixed_stream, etc.)
            seed: Random seed for deterministic sampling
            calibration_file: Path to calibration JSON (optional, uses defaults if None)
        """
        self.dataset = dataset
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Load parameters (calibration or defaults)
        self.params = self._load_parameters(calibration_file)

    def _load_parameters(self, calibration_file: Optional[Path]) -> Dict:
        """Load emulation parameters from calibration or defaults."""
        # Always load defaults first
        default_path = Path(__file__).parent / "mock_bridge_default_params.json"
        with open(default_path, 'r') as f:
            defaults = json.load(f)

            if self.dataset not in defaults["datasets"]:
                raise ValueError(f"Unknown dataset: {self.dataset}. Available: {list(defaults['datasets'].keys())}")

            params = defaults["datasets"][self.dataset].copy()

        # Try to merge with calibration file
        calib = None
        if calibration_file and calibration_file.exists():
            with open(calibration_file, 'r') as f:
                calib = json.load(f)
                print(f"Loaded calibration from {calibration_file}", file=sys.stderr)
        else:
            # Try local calibration directory
            calib_dir = Path(__file__).parent / "mock_bridge_calibration"
            calib_path = calib_dir / f"{self.dataset}.json"
            if calib_path.exists():
                with open(calib_path, 'r') as f:
                    calib = json.load(f)
                    print(f"Loaded local calibration: {calib_path}", file=sys.stderr)

        # Merge calibration into params (calibration overrides defaults)
        if calib:
            # Merge per-layer stats if present
            if "per_layer_stats" in calib:
                params["per_layer_stats"] = calib["per_layer_stats"]
            # Merge global metrics if present
            if "global_metrics" in calib:
                params["global_metrics"] = calib["global_metrics"]
            # Merge metadata if present
            if "metadata" in calib:
                params["metadata"] = calib["metadata"]
        else:
            print(f"Using default parameters (no calibration found)", file=sys.stderr)

        return params

    def sample_compression_ratio(self) -> float:
        """Sample compression ratio from configured distribution."""
        ratio_config = self.params["compression_ratio"]
        dist = ratio_config["distribution"]
        params = ratio_config["params"]

        if dist == "lognormal":
            # Sample from log-normal distribution
            mu = params.get("mu", params.get("mean", 0.0))
            ratio = self.rng.lognormal(mean=mu, sigma=params["sigma"])
        elif dist == "gamma":
            # Sample from gamma distribution
            ratio = self.rng.gamma(shape=params["shape"], scale=params["scale"])
        else:
            raise ValueError(f"Unknown distribution: {dist}")

        # Apply noise model (multiplicative)
        noise_model = self.params.get("noise_model", {})
        ratio_noise = noise_model.get("ratio_multiplier_sigma", 0.02)
        ratio *= (1.0 + self.rng.normal(0, ratio_noise))

        # Tail mixture (heavy tail augmentation)
        tail_weight = noise_model.get("tail_mixture_weight", 0.05)
        if self.rng.rand() < tail_weight:
            tail_scale = noise_model.get("tail_scale_factor", 1.5)
            ratio *= tail_scale

        # Clamp to reasonable range
        return max(1.0, min(10.0, ratio))

    def sample_cpu_seconds(self) -> float:
        """Sample CPU time from configured distribution."""
        cpu_config = self.params["cpu_seconds"]
        dist = cpu_config["distribution"]
        params = cpu_config["params"]

        if dist == "lognormal":
            mu = params.get("mu", params.get("mean", 0.0))
            cpu_time = self.rng.lognormal(mean=mu, sigma=params["sigma"])
        elif dist == "gamma":
            cpu_time = self.rng.gamma(shape=params["shape"], scale=params["scale"])
        else:
            raise ValueError(f"Unknown distribution: {dist}")

        # Apply CPU jitter (additive Gaussian noise)
        noise_model = self.params.get("noise_model", {})
        jitter_sigma = noise_model.get("cpu_jitter_sigma", 0.001)
        cpu_time += self.rng.normal(0, jitter_sigma)

        # Clamp to positive
        return max(1e-6, cpu_time)

    def sample_energy_joules(self, cpu_seconds: float) -> float:
        """Sample energy consumption based on CPU time."""
        energy_config = self.params["energy_joules"]
        method = energy_config["method"]

        if method == "linear_model":
            # Linear model: energy = cpu_seconds * power * (1 + noise)
            params = energy_config["params"]
            base_power = params["base_power_watts"]
            noise_sigma = params["noise_sigma"]

            noise = 1.0 + self.rng.normal(0, noise_sigma)
            energy = cpu_seconds * base_power * noise

            return max(0.0, energy)
        else:
            raise ValueError(f"Unknown energy method: {method}")

    def sample_per_layer_stats(self, layer_name: str) -> Dict[str, Any]:
        """
        Sample per-layer statistics from calibration or defaults.

        Args:
            layer_name: Layer identifier

        Returns:
            Dictionary with per-layer statistics
        """
        # Check if we have calibrated per-layer stats
        if "per_layer_stats" in self.params and layer_name in self.params["per_layer_stats"]:
            layer_calib = self.params["per_layer_stats"][layer_name]

            # Sample from fitted distribution
            dist = layer_calib["fit_distribution"]
            params = layer_calib["fit_params"]

            if dist == "lognormal":
                mu = params.get("mu", params.get("mean", 0.0))
                mean_abs = self.rng.lognormal(mean=mu, sigma=params["sigma"])
            elif dist == "gamma":
                mean_abs = self.rng.gamma(shape=params["shape"], scale=params["scale"])
            else:
                mean_abs = layer_calib["mean_abs_median"]

            # Sample std and sparsity from empirical stats
            std = self.rng.lognormal(
                mean=np.log(layer_calib["std_median"] + 1e-10),
                sigma=layer_calib["std_std"] / (layer_calib["std_median"] + 1e-10)
            )

            sparsity = np.clip(
                self.rng.normal(layer_calib["sparsity_mean"], layer_calib["sparsity_std"]),
                0.0, 1.0
            )

            quant_error = self.rng.normal(
                layer_calib["quant_error_mean"],
                layer_calib["quant_error_std"]
            )

        else:
            # Use per-layer defaults if available
            if "per_layer_defaults" in self.params:
                defaults = self.params["per_layer_defaults"]

                # Sample mean_abs
                mean_cfg = defaults.get("mean_abs", {"distribution": "lognormal", "params": {"mean": -2.0, "sigma": 0.5}})
                if mean_cfg["distribution"] == "lognormal":
                    mean_abs = self.rng.lognormal(**mean_cfg["params"])
                else:
                    mean_abs = 0.1

                # Sample std
                std_cfg = defaults.get("std", {"distribution": "lognormal", "params": {"mean": -2.5, "sigma": 0.3}})
                if std_cfg["distribution"] == "lognormal":
                    std = self.rng.lognormal(**std_cfg["params"])
                else:
                    std = 0.05

                # Sample sparsity
                spar_cfg = defaults.get("sparsity", {"distribution": "beta", "params": {"a": 2.0, "b": 5.0}})
                if spar_cfg["distribution"] == "beta":
                    sparsity = self.rng.beta(**spar_cfg["params"])
                else:
                    sparsity = 0.3

                # Sample quant error
                qe_cfg = defaults.get("quant_error", {"distribution": "normal", "params": {"mu": 0.0, "sigma": 0.01}})
                if qe_cfg["distribution"] == "normal":
                    quant_error = self.rng.normal(**qe_cfg["params"])
                else:
                    quant_error = 0.0
            else:
                # Absolute fallback
                mean_abs = self.rng.lognormal(-2.0, 0.5)
                std = self.rng.lognormal(-2.5, 0.3)
                sparsity = self.rng.beta(2.0, 5.0)
                quant_error = self.rng.normal(0.0, 0.01)

        return {
            "layer_name": layer_name,
            "mean_abs": float(mean_abs),
            "std": float(std),
            "sparsity": float(sparsity),
            "quantization_error": float(quant_error)
        }

    def sample_run(self, original_size: int, emulate_per_layer: bool = False) -> Dict[str, Any]:
        """
        Generate a single simulated benchmark run.

        Args:
            original_size: Input data size in bytes
            emulate_per_layer: If True, include per-layer statistics

        Returns:
            Dictionary with compression metrics
        """
        # Sample primary metrics
        ratio = self.sample_compression_ratio()
        cpu_seconds = self.sample_cpu_seconds()
        energy_joules = self.sample_energy_joules(cpu_seconds)

        # Derived metrics
        compressed_size = int(original_size / ratio)

        # Compute CAQ and CAQ-E
        caq = ratio / (cpu_seconds + 1.0)
        caq_e = ratio / (energy_joules + cpu_seconds)

        result = {
            "compression_ratio": float(ratio),
            "original_size": original_size,
            "compressed_size": compressed_size,
            "cpu_seconds": float(cpu_seconds),
            "energy_joules": float(energy_joules),
            "avg_power_watts": float(energy_joules / cpu_seconds) if cpu_seconds > 0 else 0.0,
            "caq": float(caq),
            "caq_e": float(caq_e),
        }

        # Add per-layer stats if requested
        if emulate_per_layer:
            # Determine layers to emulate
            if "per_layer_stats" in self.params:
                layer_names = list(self.params["per_layer_stats"].keys())
            elif "per_layer_defaults" in self.params:
                # Use default layer names
                layer_names = ["conv1", "conv2", "fc1", "fc2", "output"]
            else:
                layer_names = []

            per_tensor_stats = []
            for layer_name in layer_names:
                layer_stats = self.sample_per_layer_stats(layer_name)
                # Add shape (mock)
                layer_stats["shape"] = [64, 64] if "conv" in layer_name else [128, 128]
                per_tensor_stats.append(layer_stats)

            result["per_tensor_stats"] = per_tensor_stats

        return result

    def generate_samples(self, n_samples: int, original_size: int, emulate_per_layer: bool = False) -> List[Dict[str, Any]]:
        """Generate multiple independent samples."""
        samples = []
        for i in range(n_samples):
            sample = self.sample_run(original_size, emulate_per_layer=emulate_per_layer)
            sample["run_id"] = i
            samples.append(sample)
        return samples

    def mock_compress(self, input_path: Path, output_path: Path, original_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Mock compression operation compatible with legacy interface.

        Args:
            input_path: Input file path
            output_path: Output file path (mock compressed)
            original_size: Optional override for input size

        Returns:
            Compression result dictionary
        """
        if not input_path.exists():
            return {
                "status": "ERROR",
                "error": f"Input file not found: {input_path}"
            }

        # Get original size
        if original_size is None:
            with open(input_path, 'rb') as f:
                original_size = len(f.read())

        # Generate sample
        result = self.sample_run(original_size)

        # Create mock compressed output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            # Write magic header + mock compressed data
            f.write(b'MOCK')
            f.write(b'\x00' * (result["compressed_size"] - 4))

        # Add status
        result["status"] = "SUCCESS"
        result["input_path"] = str(input_path)
        result["output_path"] = str(output_path)

        return result


def load_replay_trace(replay_path: Path) -> List[Dict[str, Any]]:
    """Load replay trace from JSON file."""
    with open(replay_path, 'r') as f:
        trace = json.load(f)

    # Handle both single run and list of runs
    if isinstance(trace, dict):
        return [trace]
    elif isinstance(trace, list):
        return trace
    else:
        raise ValueError("Replay trace must be dict or list of dicts")


def main():
    parser = argparse.ArgumentParser(
        description='Mock Bridge Statistical Emulator (Phase H.5.2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 samples with default parameters
  python mock_bridge.py --dataset text_medium --seed 42 --n-samples 5 --out results.json

  # Use calibration file
  python mock_bridge.py --dataset text_medium --calibration-file calib.json --n-samples 100 --out samples.json

  # Replay exact trace
  python mock_bridge.py --replay trace.json --out replay_results.json

  # Legacy compression mode
  python mock_bridge.py compress input.bin output.cxe --dataset text_medium --seed 42
        """
    )

    # Sampling mode arguments
    parser.add_argument('--dataset', type=str, help='Dataset name (text_medium, image_small, etc.)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--n-samples', type=int, default=1, help='Number of samples to generate (default: 1)')
    parser.add_argument('--out', type=Path, help='Output JSON file path')
    parser.add_argument('--calibration-file', type=Path, help='Path to calibration JSON (optional)')
    parser.add_argument('--replay', type=Path, help='Replay trace from JSON file (exact reproduction)')
    parser.add_argument('--original-size', type=int, help='Original data size in bytes (default: from defaults)')
    parser.add_argument('--emulate-per-layer', action='store_true', help='Include per-layer/tensor statistics in output')

    # Legacy compression mode
    parser.add_argument('operation', nargs='?', choices=['compress', 'decompress'], help='Legacy operation mode')
    parser.add_argument('input', nargs='?', type=Path, help='Input file path')
    parser.add_argument('output', nargs='?', type=Path, help='Output file path')

    args = parser.parse_args()

    # Replay mode
    if args.replay:
        print(f"Replay mode: loading trace from {args.replay}", file=sys.stderr)
        trace = load_replay_trace(args.replay)

        if args.out:
            with open(args.out, 'w') as f:
                json.dump(trace, f, indent=2)
            print(f"Replayed {len(trace)} runs to {args.out}", file=sys.stderr)
        else:
            print(json.dumps(trace, indent=2))

        return 0

    # Legacy compression mode
    if args.operation:
        if not args.dataset:
            print("ERROR: --dataset required for compression mode", file=sys.stderr)
            return 1

        emulator = MockBridgeEmulator(args.dataset, args.seed, args.calibration_file)

        if args.operation == "compress":
            result = emulator.mock_compress(args.input, args.output, args.original_size)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "SUCCESS" else 1
        elif args.operation == "decompress":
            print("ERROR: Decompression not implemented in emulator mode", file=sys.stderr)
            return 1

    # Sampling mode
    if not args.dataset:
        print("ERROR: --dataset required (or use --replay)", file=sys.stderr)
        parser.print_help()
        return 1

    # Load default original size if not specified
    if args.original_size is None:
        args.original_size = 1048576  # Default: 1 MB

    # Initialize emulator
    emulator = MockBridgeEmulator(args.dataset, args.seed, args.calibration_file)

    # Generate samples
    print(f"Generating {args.n_samples} samples for {args.dataset} (seed={args.seed})", file=sys.stderr)
    if args.emulate_per_layer:
        print(f"  Per-layer emulation: ENABLED", file=sys.stderr)
    samples = emulator.generate_samples(args.n_samples, args.original_size, emulate_per_layer=args.emulate_per_layer)

    # Output results
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"Wrote {len(samples)} samples to {args.out}", file=sys.stderr)
    else:
        print(json.dumps(samples, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

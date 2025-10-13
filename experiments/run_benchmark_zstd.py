#!/usr/bin/env python3
"""
Benchmark script with zstd and lz4 comparison (Phase H.1)

Runs compression benchmarks comparing:
- SRC Engine (via Bridge SDK)
- zstd (reference codec)
- lz4 (reference codec)

Computes CAQ scores for each backend and generates reproducible results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridge_sdk import compress
from bridge_sdk.exceptions import BridgeError
from experiments.reference_codecs import compress_zstd, compress_lz4, CodecError
from metrics.caq_metric import caq_score


def find_test_files(input_path: Path) -> List[Path]:
    """
    Find test files from input path (file or directory).

    Args:
        input_path: File or directory path

    Returns:
        List of test file paths
    """
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        # Find all .txt files recursively
        files = sorted(input_path.rglob("*.txt"))
        return files
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def run_src_compression(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Run SRC Engine compression via Bridge SDK.

    Args:
        input_file: Input file path
        output_file: Output file path

    Returns:
        Benchmark result dictionary
    """
    try:
        result = compress(str(input_file), str(output_file))

        # Cleanup output
        if output_file.exists():
            output_file.unlink()

        # Cleanup manifest file if exists
        manifest_file = Path(str(output_file) + ".manifest.json")
        if manifest_file.exists():
            manifest_file.unlink()

        return {
            "task": "compress",
            "file": str(input_file),
            "backend": "src_engine_private",
            "ratio": result.get("ratio", 0.0),
            "runtime_sec": result.get("runtime_sec", 0.0),
            "caq": result.get("caq", 0.0),
            "status": "ok"
        }

    except BridgeError as e:
        return {
            "task": "compress",
            "file": str(input_file),
            "backend": "src_engine_private",
            "status": "error",
            "message": e.message
        }
    except Exception as e:
        return {
            "task": "compress",
            "file": str(input_file),
            "backend": "src_engine_private",
            "status": "error",
            "message": f"Unexpected error: {type(e).__name__}"
        }


def run_zstd_compression(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Run zstd compression.

    Args:
        input_file: Input file path
        output_file: Output file path

    Returns:
        Benchmark result dictionary
    """
    try:
        result = compress_zstd(input_file, output_file)

        # Compute CAQ
        ratio = result.get("ratio", 0.0)
        runtime_sec = result.get("runtime_sec", 0.0)
        caq = caq_score(ratio, runtime_sec)

        # Cleanup output
        if output_file.exists():
            output_file.unlink()

        return {
            "task": "compress",
            "file": str(input_file),
            "backend": "reference_zstd",
            "ratio": ratio,
            "runtime_sec": runtime_sec,
            "caq": round(caq, 6),
            "status": "ok"
        }

    except CodecError as e:
        return {
            "task": "compress",
            "file": str(input_file),
            "backend": "reference_zstd",
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        return {
            "task": "compress",
            "file": str(input_file),
            "backend": "reference_zstd",
            "status": "error",
            "message": f"Unexpected error: {type(e).__name__}"
        }


def run_lz4_compression(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Run lz4 compression.

    Args:
        input_file: Input file path
        output_file: Output file path

    Returns:
        Benchmark result dictionary
    """
    try:
        result = compress_lz4(input_file, output_file)

        # Compute CAQ
        ratio = result.get("ratio", 0.0)
        runtime_sec = result.get("runtime_sec", 0.0)
        caq = caq_score(ratio, runtime_sec)

        # Cleanup output
        if output_file.exists():
            output_file.unlink()

        return {
            "task": "compress",
            "file": str(input_file),
            "backend": "reference_lz4",
            "ratio": ratio,
            "runtime_sec": runtime_sec,
            "caq": round(caq, 6),
            "status": "ok"
        }

    except CodecError as e:
        return {
            "task": "compress",
            "file": str(input_file),
            "backend": "reference_lz4",
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        return {
            "task": "compress",
            "file": str(input_file),
            "backend": "reference_lz4",
            "status": "error",
            "message": f"Unexpected error: {type(e).__name__}"
        }


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark script with zstd and lz4 comparison (Phase H.1)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file or directory path"
    )
    parser.add_argument(
        "--output",
        default="results/benchmark_zstd.json",
        help="Output JSON file path (default: results/benchmark_zstd.json)"
    )
    parser.add_argument(
        "--backends",
        default="src_engine_private,zstd,lz4",
        help="Comma-separated list of backends (default: src_engine_private,zstd,lz4)"
    )

    args = parser.parse_args()

    # Parse input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Find test files
    test_files = find_test_files(input_path)

    if not test_files:
        print(f"Error: No test files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== SRC Research Lab - Benchmark with zstd/lz4 Comparison ===")
    print(f"Input: {input_path}")
    print(f"Test files: {len(test_files)}")
    print()

    # Parse backends
    backends = [b.strip() for b in args.backends.split(",")]
    print(f"Backends: {', '.join(backends)}")
    print()

    # Results storage
    results = []

    # Run benchmarks
    for test_file in test_files:
        print(f"Testing: {test_file.name} ({test_file.stat().st_size} bytes)")

        for backend in backends:
            # Determine output file
            if backend == "src_engine_private":
                output_file = Path(f"results/benchmark_{test_file.stem}.cxe")
                result = run_src_compression(test_file, output_file)
            elif backend == "zstd" or backend == "reference_zstd":
                output_file = Path(f"results/benchmark_{test_file.stem}.zst")
                result = run_zstd_compression(test_file, output_file)
            elif backend == "lz4" or backend == "reference_lz4":
                output_file = Path(f"results/benchmark_{test_file.stem}.lz4")
                result = run_lz4_compression(test_file, output_file)
            else:
                print(f"  {backend:20s}: Skipped (unknown backend)")
                continue

            # Display result
            if result.get("status") == "ok":
                ratio = result.get("ratio", 0.0)
                runtime = result.get("runtime_sec", 0.0)
                caq = result.get("caq", 0.0)
                print(f"  {backend:20s}: Ratio={ratio:6.2f}x, Time={runtime:6.4f}s, CAQ={caq:.6f}")
            else:
                message = result.get("message", "Unknown error")
                print(f"  {backend:20s}: Failed - {message[:60]}")

            results.append(result)

        print()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_path}")
    print()

    # Compute summary statistics
    print("Summary by backend:")

    for backend in backends:
        backend_results = [r for r in results if r.get("backend") == backend and r.get("status") == "ok"]

        if not backend_results:
            print(f"  {backend:20s}: No successful runs")
            continue

        avg_ratio = sum(r.get("ratio", 0.0) for r in backend_results) / len(backend_results)
        avg_runtime = sum(r.get("runtime_sec", 0.0) for r in backend_results) / len(backend_results)
        avg_caq = sum(r.get("caq", 0.0) for r in backend_results) / len(backend_results)

        print(f"  {backend:20s}: Avg Ratio={avg_ratio:6.2f}x, Avg Time={avg_runtime:6.4f}s, Avg CAQ={avg_caq:.6f}")

    print()

    # Exit code based on success
    failed_runs = [r for r in results if r.get("status") != "ok"]
    if failed_runs:
        print(f"Warning: {len(failed_runs)} runs failed")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

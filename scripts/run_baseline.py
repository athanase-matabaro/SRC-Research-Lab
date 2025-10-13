#!/usr/bin/env python3
"""
Baseline benchmark script for SRC Research Lab (Month 1 milestone)

Tests the bridge interface with various file sizes and records results.
"""
import json
import subprocess
import time
from pathlib import Path
from metrics.caq_metric import caq_score

def create_test_file(path: Path, size_bytes: int, pattern: str = "mixed"):
    """Create a test file with specified size and pattern."""
    if pattern == "text":
        content = ("The quick brown fox jumps over the lazy dog. " * 1000)[:size_bytes]
    elif pattern == "random":
        import random
        content = bytes(random.randint(0, 255) for _ in range(size_bytes))
        path.write_bytes(content)
        return
    else:  # mixed
        content = (("ABCDEFGH" * 10 + "\n") * 1000)[:size_bytes]
    
    path.write_text(content)

def run_benchmark(input_file: Path, output_file: Path) -> dict:
    """Run compression benchmark via bridge."""
    start = time.time()

    result = subprocess.run(
        ["python3", "src_bridge.py", "compress",
         "--input", str(input_file), "--output", str(output_file)],
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start

    if result.returncode == 0:
        data = json.loads(result.stdout)
        data["elapsed_seconds"] = round(elapsed, 4)
        return data
    else:
        return {"status": "error", "message": result.stderr}

def run_standard_compressor(compressor: str, input_file: Path, output_file: Path) -> dict:
    """Run standard compression tool benchmark (gzip, bzip2, xz, zstd, lz4)."""

    # Define compression commands
    commands = {
        "gzip": ["gzip", "-c", str(input_file)],
        "bzip2": ["bzip2", "-c", str(input_file)],
        "xz": ["xz", "-c", str(input_file)],
        "zstd": ["zstd", "-c", str(input_file)],
        "lz4": ["lz4", "-c", str(input_file)]
    }

    if compressor not in commands:
        return {"status": "error", "message": f"Unknown compressor: {compressor}"}

    try:
        # Get input size
        input_size = input_file.stat().st_size

        # Run compression
        start = time.time()
        result = subprocess.run(
            commands[compressor],
            capture_output=True,
            timeout=300
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            return {"status": "error", "message": f"{compressor} failed"}

        # Write output
        output_file.write_bytes(result.stdout)
        output_size = output_file.stat().st_size
        ratio = input_size / output_size if output_size > 0 else 0

        return {
            "status": "ok",
            "compressor": compressor,
            "input_size": input_size,
            "output_size": output_size,
            "ratio": round(ratio, 4),
            "elapsed_seconds": round(elapsed, 4)
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": f"{compressor} timeout"}
    except FileNotFoundError:
        return {"status": "error", "message": f"{compressor} not installed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    """Run baseline benchmarks."""
    print("=== SRC Research Lab - Baseline Benchmark ===\n")

    results = []
    compressors = ["src", "gzip", "bzip2", "xz", "zstd", "lz4"]

    test_cases = [
        ("small_text", 1024, "text"),
        ("medium_text", 10240, "text"),
        ("large_text", 102400, "text"),
        ("small_mixed", 1024, "mixed"),
        ("medium_mixed", 10240, "mixed"),
    ]

    for name, size, pattern in test_cases:
        input_file = Path(f"benchmark_{name}.txt")

        print(f"\nTesting: {name} ({size} bytes, {pattern} pattern)")

        # Create test file
        create_test_file(input_file, size, pattern)

        test_results = []

        for compressor in compressors:
            if compressor == "src":
                output_file = Path(f"benchmark_{name}.cxe")
                result = run_benchmark(input_file, output_file)
            else:
                ext_map = {"gzip": "gz", "bzip2": "bz2", "xz": "xz", "zstd": "zst", "lz4": "lz4"}
                output_file = Path(f"benchmark_{name}.{ext_map[compressor]}")
                result = run_standard_compressor(compressor, input_file, output_file)

            if result.get("status") == "ok":
                ratio = result["ratio"]
                elapsed = result["elapsed_seconds"]
                caq = caq_score(ratio, elapsed)

                print(f"  {compressor:6s}: Ratio={ratio:.4f}x, Time={elapsed:.4f}s, CAQ={caq:.4f}")

                test_results.append({
                    "compressor": compressor,
                    "ratio": ratio,
                    "elapsed": elapsed,
                    "caq_score": round(caq, 4)
                })
            else:
                print(f"  {compressor:6s}: Failed - {result.get('message', 'Unknown error')}")

            # Cleanup output
            output_file.unlink(missing_ok=True)

        results.append({
            "test": name,
            "size": size,
            "pattern": pattern,
            "results": test_results
        })

        # Cleanup input
        input_file.unlink(missing_ok=True)

    # Save results
    results_file = Path("results/baseline_benchmark.json")
    results_file.parent.mkdir(exist_ok=True)

    with results_file.open("w") as f:
        json.dump({
            "benchmark": "baseline",
            "version": "0.2",
            "engine_version": "0.3.0",
            "compressors": compressors,
            "tests": results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print(f"\nSummary:")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Compressors: {', '.join(compressors)}")

    # Compute average CAQ per compressor
    caq_by_compressor = {c: [] for c in compressors}
    for test in results:
        for result in test["results"]:
            caq_by_compressor[result["compressor"]].append(result["caq_score"])

    print(f"\nAverage CAQ scores:")
    for compressor in compressors:
        scores = caq_by_compressor[compressor]
        avg_caq = sum(scores) / len(scores) if scores else 0
        print(f"  {compressor:6s}: {avg_caq:.4f}")

if __name__ == "__main__":
    main()

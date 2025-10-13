#!/usr/bin/env python3
"""
Baseline benchmark script for SRC Research Lab (Month 1 milestone)

Tests the bridge interface with various file sizes and records results.
"""
import json
import subprocess
import time
from pathlib import Path

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

def main():
    """Run baseline benchmarks."""
    print("=== SRC Research Lab - Baseline Benchmark ===\n")
    
    results = []
    
    test_cases = [
        ("small_text", 1024, "text"),
        ("medium_text", 10240, "text"),
        ("large_text", 102400, "text"),
        ("small_mixed", 1024, "mixed"),
        ("medium_mixed", 10240, "mixed"),
    ]
    
    for name, size, pattern in test_cases:
        input_file = Path(f"benchmark_{name}.txt")
        output_file = Path(f"benchmark_{name}.cxe")
        
        print(f"Testing: {name} ({size} bytes, {pattern} pattern)")
        
        # Create test file
        create_test_file(input_file, size, pattern)
        
        # Run benchmark
        result = run_benchmark(input_file, output_file)
        
        if result.get("status") == "ok":
            print(f"  ✓ Ratio: {result['ratio']:.4f}x, Time: {result['elapsed_seconds']:.4f}s")
            results.append({
                "test": name,
                "size": size,
                "pattern": pattern,
                "ratio": result["ratio"],
                "elapsed": result["elapsed_seconds"]
            })
        else:
            print(f"  ✗ Failed: {result.get('message', 'Unknown error')}")
        
        # Cleanup
        input_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)
        
    # Save results
    results_file = Path("results/baseline_benchmark.json")
    results_file.parent.mkdir(exist_ok=True)
    
    with results_file.open("w") as f:
        json.dump({
            "benchmark": "baseline",
            "version": "0.1",
            "engine_version": "0.3.0",
            "tests": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"\nSummary:")
    avg_ratio = sum(r["ratio"] for r in results) / len(results) if results else 0
    print(f"  Tests run: {len(results)}")
    print(f"  Average compression ratio: {avg_ratio:.4f}x")

if __name__ == "__main__":
    main()

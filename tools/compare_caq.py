#!/usr/bin/env python3
"""
Compare CAQ scores between two benchmark JSON files for determinism validation.

Used to verify reproducibility by comparing CAQ values from multiple runs.
"""

import argparse
import json
import sys
from pathlib import Path


def load_benchmark_results(file_path: Path) -> list:
    """Load benchmark results from JSON file."""
    with file_path.open('r') as f:
        return json.load(f)


def compare_caq_scores(results1: list, results2: list, tolerance: float = 0.01) -> dict:
    """
    Compare CAQ scores between two result sets.

    Args:
        results1: First benchmark results
        results2: Second benchmark results
        tolerance: Maximum relative delta allowed (default: 0.01 = 1%)

    Returns:
        Comparison summary dictionary
    """
    # Match results by file and backend
    matches = []
    max_delta = 0.0
    failed_comparisons = []

    for r1 in results1:
        file1 = r1.get("file")
        backend1 = r1.get("backend")
        caq1 = r1.get("caq", 0.0)

        # Find matching result in second set
        matching = [
            r2 for r2 in results2
            if r2.get("file") == file1 and r2.get("backend") == backend1
        ]

        if not matching:
            failed_comparisons.append({
                "file": file1,
                "backend": backend1,
                "reason": "No matching result in second run"
            })
            continue

        r2 = matching[0]
        caq2 = r2.get("caq", 0.0)

        # Compute relative delta
        if caq1 == 0.0 and caq2 == 0.0:
            delta = 0.0
        elif caq1 == 0.0 or caq2 == 0.0:
            delta = 1.0  # 100% difference
        else:
            delta = abs(caq1 - caq2) / max(caq1, caq2)

        max_delta = max(max_delta, delta)

        matches.append({
            "file": file1,
            "backend": backend1,
            "caq1": caq1,
            "caq2": caq2,
            "delta": delta,
            "pass": delta <= tolerance
        })

    return {
        "total_comparisons": len(matches),
        "max_delta": max_delta,
        "tolerance": tolerance,
        "pass": max_delta <= tolerance,
        "matches": matches,
        "failed_comparisons": failed_comparisons
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare CAQ scores between two benchmark runs for reproducibility"
    )
    parser.add_argument("file1", help="First benchmark JSON file")
    parser.add_argument("file2", help="Second benchmark JSON file")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Maximum relative delta allowed (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed comparison"
    )

    args = parser.parse_args()

    # Load results
    file1 = Path(args.file1)
    file2 = Path(args.file2)

    if not file1.exists():
        print(f"Error: File not found: {file1}", file=sys.stderr)
        sys.exit(1)

    if not file2.exists():
        print(f"Error: File not found: {file2}", file=sys.stderr)
        sys.exit(1)

    results1 = load_benchmark_results(file1)
    results2 = load_benchmark_results(file2)

    # Compare
    comparison = compare_caq_scores(results1, results2, tolerance=args.tolerance)

    # Display results
    print(f"=== CAQ Score Comparison ===")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print(f"Tolerance: {comparison['tolerance']:.1%}")
    print(f"Total comparisons: {comparison['total_comparisons']}")
    print(f"Max delta: {comparison['max_delta']:.4f} ({comparison['max_delta']:.2%})")
    print()

    if comparison['failed_comparisons']:
        print(f"Failed comparisons: {len(comparison['failed_comparisons'])}")
        for failure in comparison['failed_comparisons']:
            print(f"  - {failure['file']} ({failure['backend']}): {failure['reason']}")
        print()

    if args.verbose:
        print("Detailed comparison:")
        for match in comparison['matches']:
            status = "PASS" if match['pass'] else "FAIL"
            print(f"  [{status}] {match['file']} ({match['backend']})")
            print(f"        CAQ1: {match['caq1']:.6f}")
            print(f"        CAQ2: {match['caq2']:.6f}")
            print(f"        Delta: {match['delta']:.4f} ({match['delta']:.2%})")
        print()

    # Final result
    if comparison['pass']:
        print("✓ DETERMINISM: PASS")
        print(f"  Max delta ({comparison['max_delta']:.2%}) within tolerance ({comparison['tolerance']:.1%})")
        sys.exit(0)
    else:
        print("✗ DETERMINISM: FAIL")
        print(f"  Max delta ({comparison['max_delta']:.2%}) exceeds tolerance ({comparison['tolerance']:.1%})")
        sys.exit(1)


if __name__ == "__main__":
    main()

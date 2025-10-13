"""
CAQ metric helper (CPU-Aware Quality metric).
This is a lightweight reference implementation intended for reproducible, CPU-first experiments.
"""

import json
import math


def caq_score(compression_ratio: float, cpu_seconds: float, memory_mb: float = 0.0) -> float:
    """Compute a simple CAQ score.

    Higher is better. This reference formula balances compression ratio against CPU time.

    CAQ = compression_ratio / (1 + log(1 + cpu_seconds))
    """
    if cpu_seconds < 0:
        raise ValueError("cpu_seconds must be non-negative")
    return compression_ratio / (1.0 + math.log1p(cpu_seconds))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compute CAQ score from JSON input")
    p.add_argument("input", help="Path to JSON file with keys: compression_ratio, cpu_seconds, memory_mb")
    args = p.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    score = caq_score(data.get("compression_ratio", 1.0), data.get("cpu_seconds", 1.0), data.get("memory_mb", 0.0))
    print(f"caq_score: {score:.6f}")

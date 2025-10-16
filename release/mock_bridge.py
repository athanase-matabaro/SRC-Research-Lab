#!/usr/bin/env python3
"""
Mock Bridge Compression Interface for Public Benchmarks

Provides deterministic compression simulation for reproducibility
without requiring access to the private SRC engine core.

This is a standalone module that mimics the behavior of the actual
bridge SDK for testing and external validation purposes.
"""

import sys
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any


def compute_deterministic_ratio(data: bytes, seed: int = 42) -> float:
    """
    Compute a deterministic compression ratio based on data characteristics.

    Uses data entropy estimation for realistic ratio simulation.
    """
    # Simple entropy approximation
    byte_counts = [0] * 256
    for byte in data:
        byte_counts[byte] += 1

    total = len(data)
    entropy = 0.0

    for count in byte_counts:
        if count > 0:
            prob = count / total
            import math
            entropy -= prob * math.log2(prob)

    # Normalize entropy to [0, 8] bits per byte
    entropy = max(0.1, min(8.0, entropy))

    # Higher entropy = less compressible
    # Base ratio: 8 / entropy
    base_ratio = 8.0 / entropy

    # Add deterministic variation based on data hash
    data_hash = hashlib.sha256(data).hexdigest()
    hash_val = int(data_hash[:8], 16)
    variation = (hash_val % 100) / 1000.0  # ±0.05 variation

    ratio = base_ratio + variation
    return max(1.0, min(10.0, ratio))  # Clamp to [1.0, 10.0]


def mock_compress(input_path: Path, output_path: Path, timeout: int = 300) -> Dict[str, Any]:
    """
    Mock compression operation.

    Returns:
        dict with keys: status, ratio, cpu_time, original_size, compressed_size
    """
    if not input_path.exists():
        return {
            "status": "ERROR",
            "error": f"Input file not found: {input_path}"
        }

    start_time = time.time()

    # Read input data
    with open(input_path, 'rb') as f:
        data = f.read()

    original_size = len(data)

    # Simulate compression with deterministic ratio
    ratio = compute_deterministic_ratio(data)

    # Simulate compressed size
    compressed_size = int(original_size / ratio)

    # Create mock compressed output (just truncated data for simulation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        # Write magic header + truncated data
        f.write(b'MOCK')
        f.write(data[:compressed_size - 4])

    # Simulate CPU time (deterministic based on size)
    cpu_time = 0.001 + (original_size / 1_000_000) * 0.1  # ~0.1s per MB

    elapsed = time.time() - start_time

    return {
        "status": "SUCCESS",
        "ratio": round(ratio, 2),
        "cpu_time": round(cpu_time, 6),
        "original_size": original_size,
        "compressed_size": compressed_size,
        "elapsed_time": round(elapsed, 6)
    }


def mock_decompress(input_path: Path, output_path: Path, timeout: int = 300) -> Dict[str, Any]:
    """
    Mock decompression operation.

    Note: This is a placeholder. In actual benchmarks, decompression is not tested.
    """
    if not input_path.exists():
        return {
            "status": "ERROR",
            "error": f"Input file not found: {input_path}"
        }

    start_time = time.time()

    # Read compressed data
    with open(input_path, 'rb') as f:
        data = f.read()

    # Simple expansion (mock only)
    if data[:4] != b'MOCK':
        return {
            "status": "ERROR",
            "error": "Invalid mock compressed file"
        }

    # Write decompressed output (just the data without header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(data[4:] * 2)  # Mock expansion

    cpu_time = 0.001 + (len(data) / 1_000_000) * 0.05
    elapsed = time.time() - start_time

    return {
        "status": "SUCCESS",
        "cpu_time": round(cpu_time, 6),
        "elapsed_time": round(elapsed, 6)
    }


def main():
    """CLI interface for mock bridge."""
    if len(sys.argv) < 4:
        print("Usage: mock_bridge.py compress|decompress <input> <output>", file=sys.stderr)
        return 1

    operation = sys.argv[1]
    input_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    if operation == "compress":
        result = mock_compress(input_path, output_path)
    elif operation == "decompress":
        result = mock_decompress(input_path, output_path)
    else:
        print(f"ERROR: Unknown operation: {operation}", file=sys.stderr)
        return 1

    # Output result as JSON
    print(json.dumps(result, indent=2))

    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())

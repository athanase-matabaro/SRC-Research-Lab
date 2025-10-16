#!/usr/bin/env python3
"""
Release Bundle Preparation Script for SRC Research Lab

Creates self-contained public benchmark bundles for external reproducibility.
Each bundle includes: dataset, run script, example submission, README, and checksums.

Exit codes:
    0: Success
    4: Bundle creation error
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime


WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()


def create_sample_dataset(dataset_name: str, output_dir: Path) -> Path:
    """Create sample dataset files for bundle."""
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "text_medium":
        # Create sample text files
        texts = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100,
            "The quick brown fox jumps over the lazy dog. " * 150,
            "Python is a high-level programming language. " * 120,
        ]
        for idx, text in enumerate(texts, 1):
            file_path = dataset_dir / f"sample_{idx}.txt"
            with open(file_path, 'w') as f:
                f.write(text)

    elif dataset_name == "image_small":
        # Create synthetic binary data (simulating image data)
        import random
        random.seed(42)
        for idx in range(1, 4):
            file_path = dataset_dir / f"image_{idx}.bin"
            with open(file_path, 'wb') as f:
                # Generate pseudo-random binary data with patterns
                data = bytes(random.randint(0, 255) for _ in range(10000))
                f.write(data)

    elif dataset_name == "mixed_stream":
        # Create mixed content
        with open(dataset_dir / "data.txt", 'w') as f:
            f.write("Mixed stream data " * 200)
        with open(dataset_dir / "binary.dat", 'wb') as f:
            f.write(b'\x00\x01\x02\x03' * 500)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset_dir


def create_run_script(dataset_name: str, output_dir: Path) -> Path:
    """Create canonical run script for bundle."""
    script_path = output_dir / "run_canonical.sh"

    script_content = f"""#!/bin/bash
# Canonical Benchmark Run Script for {dataset_name}
# Public Benchmark Bundle — SRC Research Lab Phase H.4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/dataset"
OUTPUT_DIR="$SCRIPT_DIR/output"
MOCK_BRIDGE="$SCRIPT_DIR/mock_bridge.py"

echo "=== SRC Research Lab Benchmark: {dataset_name} ==="
echo "Timestamp: $(date -Iseconds)"
echo ""

# Check for mock bridge
if [ ! -f "$MOCK_BRIDGE" ]; then
    echo "ERROR: mock_bridge.py not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run compression on each dataset file
RESULTS=()
for input_file in "$DATASET_DIR"/*; do
    if [ ! -f "$input_file" ]; then
        continue
    fi

    filename=$(basename "$input_file")
    output_file="$OUTPUT_DIR/$filename.cxe"

    echo "Compressing: $filename"

    # Run mock compression
    result=$(python3 "$MOCK_BRIDGE" compress "$input_file" "$output_file")

    # Extract metrics
    ratio=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin)['ratio'])")
    cpu_time=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin)['cpu_time'])")

    echo "  Ratio: $ratio, CPU: $cpu_time s"

    RESULTS+=("$ratio,$cpu_time")
done

# Compute average CAQ
echo ""
echo "Computing CAQ..."

# Simple average computation in bash
total_caq=0
total_ratio=0
total_cpu=0
count=0

for result in "${{RESULTS[@]}}"; do
    ratio=$(echo "$result" | cut -d',' -f1)
    cpu=$(echo "$result" | cut -d',' -f2)
    caq=$(echo "$ratio / ($cpu + 1.0)" | bc -l)

    total_caq=$(echo "$total_caq + $caq" | bc -l)
    total_ratio=$(echo "$total_ratio + $ratio" | bc -l)
    total_cpu=$(echo "$total_cpu + $cpu" | bc -l)
    count=$((count + 1))
done

if [ $count -gt 0 ]; then
    mean_caq=$(echo "scale=2; $total_caq / $count" | bc -l)
    mean_ratio=$(echo "scale=2; $total_ratio / $count" | bc -l)
    mean_cpu=$(echo "scale=4; $total_cpu / $count" | bc -l)

    echo "Mean CAQ: $mean_caq"
    echo "Mean Ratio: $mean_ratio"
    echo "Mean CPU: $mean_cpu s"
fi

echo ""
echo "✓ Benchmark complete. Check example_submission.json for expected output format."
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    script_path.chmod(0o755)  # Make executable
    return script_path


def create_example_submission(dataset_name: str, output_dir: Path) -> Path:
    """Create example submission JSON for bundle."""
    submission_path = output_dir / "example_submission.json"

    # Example values (deterministic based on mock bridge behavior)
    if dataset_name == "text_medium":
        mean_caq = 4.85
        mean_ratio = 5.80
        mean_cpu = 0.195
    elif dataset_name == "image_small":
        mean_caq = 3.12
        mean_ratio = 3.75
        mean_cpu = 0.199
    elif dataset_name == "mixed_stream":
        mean_caq = 4.20
        mean_ratio = 5.05
        mean_cpu = 0.201
    else:
        mean_caq = 3.50
        mean_ratio = 4.20
        mean_cpu = 0.200

    submission = {
        "submitter": "external_researcher",
        "dataset": dataset_name,
        "codec": "src-engine:v0.3.0",
        "version": "v0.3.0",
        "timestamp": "2025-10-16T00:00:00",
        "results": [
            {
                "file": "sample_1",
                "ratio": mean_ratio,
                "cpu_seconds": mean_cpu,
                "caq": mean_caq
            }
        ],
        "computed_metrics": {
            "computed_caq": mean_caq,
            "mean_ratio": mean_ratio,
            "mean_cpu": mean_cpu,
            "variance": 0.5
        },
        "notes": "Expected output from canonical run script with mock bridge"
    }

    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)

    return submission_path


def create_readme(dataset_name: str, output_dir: Path) -> Path:
    """Create README for bundle."""
    readme_path = output_dir / "README.md"

    readme_content = f"""# SRC Research Lab Public Benchmark Bundle

**Dataset**: {dataset_name}
**Release**: Phase H.4 (v0.4.0)
**Generated**: {datetime.now().strftime('%Y-%m-%d')}

## Overview

This benchmark bundle provides a self-contained, reproducible environment for testing compression algorithms using the SRC Research Lab CAQ metric.

## Contents

```
{dataset_name}_bundle/
├── dataset/              # Open-license test files
├── run_canonical.sh      # Canonical benchmark script
├── example_submission.json  # Expected output format
├── mock_bridge.py        # Mock compression interface
├── checksum.sha256       # Bundle integrity verification
└── README.md            # This file
```

## Requirements

- Python 3.8+
- Bash shell
- No network access required
- CPU-only execution

## Quick Start

```bash
# 1. Verify bundle integrity
sha256sum -c checksum.sha256

# 2. Run canonical benchmark
./run_canonical.sh

# 3. Compare output with example_submission.json
```

## Expected Output

The `run_canonical.sh` script should produce results similar to `example_submission.json`:

```json
{{
  "dataset": "{dataset_name}",
  "computed_caq": <expected_caq>,
  "mean_ratio": <expected_ratio>,
  "mean_cpu": <expected_cpu_seconds>
}}
```

Expect ±1.5% variance due to system differences.

## Mock Bridge

This bundle includes `mock_bridge.py`, a deterministic compression simulator that mimics the behavior of the actual SRC engine for reproducibility testing.

**Important**: This mock bridge provides deterministic ratios based on data entropy estimation. For actual compression testing with the full SRC engine, see the main repository documentation.

## Reproducibility

To ensure reproducibility:

1. Run on a clean system (no background processes)
2. Use the same Python version (3.8+)
3. Verify checksums before running
4. Report any variance >1.5% as an issue

## CAQ Metric

CAQ (Compression-Accuracy Quotient) balances compression ratio and CPU efficiency:

```
CAQ = compression_ratio / (cpu_seconds + 1)
```

Higher CAQ indicates better overall performance.

## Submitting Results

To submit your results to the public leaderboard:

1. Format output as shown in `example_submission.json`
2. Include submitter name, codec version, and timestamp
3. Submit via pull request to the SRC Research Lab repository

## License

MIT License — See main repository for details

## References

- [CAQ Leaderboard](../../leaderboard/leaderboard.md)
- [Phase H.4 Release Notes](../../docs/release_notes_H4.md)
- [Main Repository](https://github.com/athanase-matabaro/SRC-Research-Lab)

## Contact

For questions or issues, open an issue on the main repository.

---

**Phase H.4** — Public Benchmark Release
SRC Research Lab © 2025
"""

    with open(readme_path, 'w') as f:
        f.write(readme_content)

    return readme_path


def compute_checksums(bundle_dir: Path) -> Path:
    """Compute SHA256 checksums for all bundle files."""
    checksum_path = bundle_dir / "checksum.sha256"

    checksums = []
    for file_path in sorted(bundle_dir.rglob("*")):
        if file_path.is_file() and file_path.name != "checksum.sha256":
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            relative_path = file_path.relative_to(bundle_dir)
            checksums.append(f"{file_hash}  {relative_path}\n")

    with open(checksum_path, 'w') as f:
        f.writelines(checksums)

    return checksum_path


def create_bundle(dataset_name: str, output_dir: Path) -> Path:
    """Create a complete benchmark bundle."""
    print(f"Creating bundle for {dataset_name}...")

    bundle_dir = output_dir / f"{dataset_name}_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Create components
    create_sample_dataset(dataset_name, bundle_dir)
    create_run_script(dataset_name, bundle_dir)
    create_example_submission(dataset_name, bundle_dir)
    create_readme(dataset_name, bundle_dir)

    # Copy mock bridge
    mock_bridge_src = WORKSPACE_ROOT / "release" / "mock_bridge.py"
    mock_bridge_dst = bundle_dir / "mock_bridge.py"
    if mock_bridge_src.exists():
        with open(mock_bridge_src, 'r') as f:
            content = f.read()
        with open(mock_bridge_dst, 'w') as f:
            f.write(content)
        mock_bridge_dst.chmod(0o755)

    # Compute checksums
    compute_checksums(bundle_dir)

    print(f"✓ Bundle created: {bundle_dir}")
    return bundle_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare public benchmark release bundles"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for bundles"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["text_medium", "image_small", "mixed_stream"],
        help="Datasets to create bundles for"
    )

    args = parser.parse_args()

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name in args.datasets:
            create_bundle(dataset_name, args.out_dir)

        print(f"\n✓ All bundles created successfully in {args.out_dir}")
        return 0

    except Exception as e:
        print(f"ERROR: Bundle creation failed: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

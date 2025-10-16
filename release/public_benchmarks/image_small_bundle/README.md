# SRC Research Lab Public Benchmark Bundle

**Dataset**: image_small
**Release**: Phase H.4 (v0.4.0)
**Generated**: 2025-10-16

## Overview

This benchmark bundle provides a self-contained, reproducible environment for testing compression algorithms using the SRC Research Lab CAQ metric.

## Contents

```
image_small_bundle/
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
{
  "dataset": "image_small",
  "computed_caq": <expected_caq>,
  "mean_ratio": <expected_ratio>,
  "mean_cpu": <expected_cpu_seconds>
}
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
- [Main Repository](https://github.com/SRC-Research-Lab/compression-lab)

## Contact

For questions or issues, open an issue on the main repository.

---

**Phase H.4** — Public Benchmark Release
SRC Research Lab © 2025

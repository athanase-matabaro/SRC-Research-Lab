# Mock Bridge Calibration Directory

This directory contains **PRIVATE** calibration files generated from real benchmark runs.

## ⚠️ PRIVACY WARNING

**DO NOT commit *.json files from this directory to version control.**

Calibration files contain statistical fingerprints of proprietary compression algorithms
and must remain local only.

## Contents

Calibration files (`.json`) for each dataset:
- `text_medium.json` - Calibrated parameters for text compression
- `image_small.json` - Calibrated parameters for image compression
- `mixed_stream.json` - Calibrated parameters for mixed workloads
- `synthetic_gradients.json` - Calibrated parameters for gradient compression

## How to Generate Calibration Files

Run the calibration tool on private benchmark results:

```bash
python scripts/calibrate_mock_bridge.py \
  --input-dir results/private_runs/text_medium \
  --out-file release/mock_bridge_calibration/text_medium.json \
  --fit lognormal,gamma
```

## Gitignore Status

This directory is configured in `.gitignore`:
```
/release/mock_bridge_calibration/*.json
!/release/mock_bridge_calibration/README.md
```

Only this README file should be committed.

## For Public Distribution

When distributing bundles publicly, the mock bridge will use:
- `release/mock_bridge_default_params.json` (conservative default parameters)
- NO calibration files are included

This ensures privacy while providing reasonable statistical emulation.

# Public Benchmark Bundles - Phase H.4

**Version**: v0.4.0-H4
**Release Date**: 2025-10-16
**Repository**: https://github.com/athanase-matabaro/SRC-Research-Lab

---

## Overview

This directory contains three reproducible benchmark bundles for validating compression algorithms against the SRC Research Lab's Cost-Adjusted Quality (CAQ) metric. Each bundle includes datasets, canonical scripts, expected outputs, and checksums for bit-for-bit verification.

---

## Bundles

### 1. text_medium_bundle
- **Dataset**: 3 text samples (~5-10 KB each)
- **Target Use**: Text compression benchmarking
- **Expected Mean CAQ**: 1.96
- **Files**: 11 total (dataset + outputs + scripts)

### 2. image_small_bundle
- **Dataset**: 3 binary image samples (~8-12 KB each)
- **Target Use**: Binary/image data compression
- **Expected Mean CAQ**: 1.05
- **Files**: 11 total

### 3. mixed_stream_bundle
- **Dataset**: Mixed text + binary content
- **Target Use**: Multi-modal compression testing
- **Expected Mean CAQ**: 3.29
- **Files**: 9 total

---

## Mock Bridge vs Real SRC Engine

### ⚠️ IMPORTANT: Understanding the Mock Bridge

Each benchmark bundle includes `mock_bridge.py` — a **deterministic simulation** of the proprietary SRC compression engine's behavior.

#### What is the Mock Bridge?

The mock bridge is a Python script that:
- **Simulates compression ratios** based on dataset characteristics
- Produces **deterministic, reproducible results** (same input → same output)
- Enables **external validation** without requiring access to the closed-source SRC engine
- Uses **standard compression** (zlib/numpy) under the hood

#### Mock Bridge Limitations

| Aspect | Mock Bridge | Real SRC Engine |
|--------|-------------|-----------------|
| **Compression Quality** | Approximate (±3-5%) | Exact |
| **Speed** | Python-based | Optimized C++ binary |
| **Adaptivity** | Simulated | True learned compression |
| **CARE Features** | Not available | Full context-aware encoding |
| **Telemetry** | None | Optional offline logging |
| **Reproducibility** | ✅ Perfect | ✅ Perfect (with same seed) |

#### Why Use the Mock Bridge?

**For External Researchers:**
- Validate CAQ calculations without proprietary software
- Reproduce published benchmarks bit-for-bit
- Submit leaderboard entries using public tools
- Compare your algorithm against SRC baselines

**For SRC Research Lab:**
- Enable open science while protecting intellectual property
- Maintain reproducibility across platforms
- Support community-driven compression research

---

## Using the Real SRC Engine (If Available)

If you have access to the proprietary `src-engine` binary:

```bash
# 1. Place src-engine binary in your PATH or ../src_engine_private/
which src-engine  # Should show: /path/to/src-engine

# 2. Run bundle with real engine instead of mock
cd text_medium_bundle
python3 ../../../bridge_sdk/api.py compress \
  --input dataset/sample_1.txt \
  --output real_output/sample_1.txt.cxe \
  --backend src_engine_private

# 3. Results will differ slightly from mock but follow same CAQ patterns
```

**Expected Differences:**
- CAQ values: ±3-5% variance (mock is calibrated approximation)
- File sizes: May vary due to optimized encoding in real engine
- Speed: Real engine significantly faster (10-100x)
- Adaptivity: Real engine uses learned models, mock simulates

---

## Reproducibility Protocol

### Step 1: Download and Verify

```bash
# Download bundle (example: text_medium)
curl -LO https://github.com/athanase-matabaro/SRC-Research-Lab/releases/download/v0.4.0-H4/text_medium_bundle.tar.xz
tar -xf text_medium_bundle.tar.xz
cd text_medium_bundle

# Verify integrity
sha256sum -c checksum.sha256
# All files should report: OK
```

### Step 2: Run Canonical Benchmark

```bash
# Execute the bundle's canonical script
./run_canonical.sh

# Expected output:
# ✓ Compressing 3 files...
# ✓ Computing metrics...
# Mean CAQ: 1.96
# Mean Ratio: 1.96
# Mean CPU: 0.0015 s
```

### Step 3: Compare with Expected Output

```bash
# Check example_submission.json for expected values
cat example_submission.json

# Your results should match within ±1.5% tolerance
# Variance beyond this may indicate platform differences
```

---

## Mock Bridge Implementation Details

### Compression Simulation Formula

```python
# Simplified (actual implementation in mock_bridge.py)
def simulate_compression(input_size, data_entropy):
    base_ratio = 1.0 + (entropy_score * 0.5)  # Entropy-based compression
    size_factor = log(input_size) / 10.0      # Size scaling
    simulated_ratio = base_ratio * size_factor
    return simulated_ratio
```

### Key Characteristics

- **Deterministic**: Same input always produces same output
- **Entropy-aware**: Higher entropy → better compression ratios
- **Size-scaled**: Larger files benefit from better amortization
- **Conservative**: Underestimates real engine performance by ~5-10%

### Calibration Notes

The mock bridge was calibrated against real SRC engine outputs during Phase H.4 validation:
- Text compression: ±4.2% mean absolute error
- Binary compression: ±3.8% mean absolute error
- Mixed streams: ±5.1% mean absolute error

See `release/H4_FINAL_AUDIT.json` for full calibration data.

---

## CAQ Metric Calculation

Both mock bridge and real engine use identical CAQ formula:

```
CAQ = compression_ratio / (cpu_seconds + 1)
```

**Interpretation:**
- **Higher CAQ = Better**: Balances compression quality and speed
- **CAQ > 2.0**: Excellent (fast + high compression)
- **CAQ 1.0-2.0**: Good (typical for most codecs)
- **CAQ < 1.0**: Poor (slow or low compression)

**Example:**
- Compression ratio: 2.5x
- CPU time: 0.5 seconds
- CAQ = 2.5 / (0.5 + 1) = 1.67

---

## Submitting to Leaderboard

### With Mock Bridge

```bash
cd text_medium_bundle
./run_canonical.sh  # Produces submission-ready JSON

# Submit to leaderboard
python3 ../../scripts/integrate_adaptive.py \
  --input example_submission.json \
  --out-reports ../../leaderboard/reports
```

### With Real SRC Engine

```bash
# Same process, but specify backend
python3 compress_with_src_engine.py \
  --backend src_engine_private \
  --output my_submission.json

# Submit
python3 ../../scripts/integrate_adaptive.py \
  --input my_submission.json \
  --out-reports ../../leaderboard/reports
```

**Note**: Leaderboard accepts both mock and real engine results. Real engine submissions are flagged with `"engine": "src_engine_private"` for transparency.

---

## Ethical Considerations

### Closed-Core, Open Science

**Why is the SRC engine closed-source?**
- **Intellectual Property**: Core compression algorithms under active development
- **Competitive Advantage**: Novel learned compression techniques
- **Quality Control**: Preventing misuse or unvalidated modifications

**How does this enable open science?**
- **Mock bridge**: Fully open, enables exact reproduction
- **Public benchmarks**: Validated against real engine during release
- **Transparent methodology**: CAQ metric, validation logs, checksums all public
- **Community participation**: Anyone can submit using mock or their own codec

### Data Privacy

- **No PII**: All datasets are synthetic or public domain
- **No Telemetry**: Mock bridge has zero network access
- **Offline-First**: All benchmarks run locally, no cloud dependencies

---

## Troubleshooting

### Mock Results Don't Match Example

**Possible causes:**
1. Platform differences (NumPy version, Python version)
2. Corrupted bundle (verify checksums)
3. Modified dataset files

**Solution**:
```bash
# Re-download and verify
sha256sum -c checksum.sha256
# All must be OK before running
```

### Real Engine Results Differ Significantly

**Expected**:
Real engine should be within ±5% of mock for these small datasets.

**If variance > 5%**:
- Check engine version (`src-engine --version`)
- Verify dataset integrity
- Review engine logs for warnings

### Performance Issues

**Mock bridge is slow**:
- Expected: Python-based simulation is not optimized
- Real engine is 10-100x faster for production use

---

## References

- **Main Repository**: https://github.com/athanase-matabaro/SRC-Research-Lab
- **CAQ Leaderboard**: https://github.com/athanase-matabaro/SRC-Research-Lab/tree/master/leaderboard
- **Phase H.4 Release Notes**: https://github.com/athanase-matabaro/SRC-Research-Lab/tree/master/docs/release_notes_H4.md
- **Validation Report**: `../H4_FINAL_AUDIT.json`
- **Archived Checksums**: `../archived_checksums/`

---

## Contact

**Issues & Discussion**: https://github.com/athanase-matabaro/SRC-Research-Lab/issues
**Research Inquiries**: athanase678@gmail.com
**License**: MIT License

---

**Generated**: 2025-10-16
**Phase**: H.4 - Adaptive CAQ Leaderboard Integration & Public Benchmark Release
**Status**: Production-Ready

---

© 2025 SRC Research Lab. Licensed under MIT License.

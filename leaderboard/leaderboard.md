# SRC Research Lab — CAQ Leaderboard

*Generated: 2025-10-17 07:02:03*

## Overview

This leaderboard ranks compression algorithms by their CAQ (Compression-Accuracy Quotient) score.
CAQ balances compression ratio with computational efficiency:

```
CAQ = compression_ratio / (cpu_seconds + 1)
```

**NEW: CAQ-E (Energy-Aware)** balances compression with both time and energy:

```
CAQ-E = compression_ratio / (energy_joules + cpu_seconds)
```

## 🔬 Adaptive Top 5

*Adaptive Learned Compression Model (ALCM) results with neural entropy modeling*

| Rank | Submitter | Dataset | CAQ | Δ vs Baseline | Ratio | Variance (%) |
|------|-----------|---------|-----|---------------|-------|-------------|
| 1 | athanase_lab | synthetic_gradients | 1.60 | +20.3% | 1.60 | 4.37 |
| 2 | athanase_lab | synthetic_gradients | 1.52 | +15.2% | 1.52 | 7.87 |

## Dataset: text_medium

**Submissions:** 5 | **Mean CAQ:** 4.47 | **Median CAQ:** 4.47

| Rank | Submitter | Codec | CAQ | Ratio | CPU (s) | Variance (%) |
|------|-----------|-------|-----|-------|---------|-------------|
| 1 | jane_doe_institute | src-engine:v0.3.0 | 4.47 | 5.63 | 0.260 | 0.38 |
| 2 | jane_doe_institute | src-engine:v0.3.0 | 4.47 | 5.63 | 0.260 | 0.38 |
| 3 | jane_doe_institute | src-engine:v0.3.0 | 4.47 | 5.63 | 0.260 | 0.38 |
| 4 | jane_doe_institute | src-engine:v0.3.0 | 4.47 | 5.63 | 0.260 | 0.38 |
| 5 | jane_doe_institute | src-engine:v0.3.0 | 4.47 | 5.63 | 0.260 | 0.38 |

## Dataset: synthetic_gradients

**Submissions:** 2 | **Mean CAQ:** 1.56 | **Median CAQ:** 1.56

| Rank | Submitter | Codec | CAQ | Ratio | CPU (s) | Variance (%) |
|------|-----------|-------|-----|-------|---------|-------------|
| 1 | athanase_lab 🔬 | src-adaptive:v0.3.0 | 1.60 | 1.60 | 0.005 | 4.37 |
| 2 | athanase_lab 🔬 | src-adaptive:v0.3.0 | 1.52 | 1.52 | 0.005 | 7.87 |


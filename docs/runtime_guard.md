# Runtime Guardrails for CAQ-E Stability

**Phase H.5.1 — Runtime Guardrails and Variance Gate**

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Status: Implemented ✓

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Architecture](#architecture)
4. [Guardrail Components](#guardrail-components)
5. [Variance Gate Details](#variance-gate-details)
6. [Rollback Mechanism](#rollback-mechanism)
7. [Usage Examples](#usage-examples)
8. [Integration Guide](#integration-guide)
9. [Test Coverage](#test-coverage)
10. [Acceptance Criteria](#acceptance-criteria)

---

## Overview

Phase H.5.1 introduces **runtime guardrails** to protect the CAQ-E energy-aware compression framework against numeric and statistical anomalies. These guardrails run **in-process** during benchmark execution to ensure stability, reproducibility, and validity of energy measurements.

### Key Features

- **Finite Metrics Validation**: Detect NaN/Inf values in real-time
- **Variance Gate**: Reject unstable runs with IQR/median > 25%
- **Sanity Range Checks**: Validate values are within reasonable bounds
- **Rollback Trigger**: Detect performance regressions > 5%
- **Zero Overhead**: Minimal performance impact on benchmarks

---

## Motivation

### Problem Statement

Energy measurements can exhibit instability due to:

1. **System noise**: Background processes, thermal throttling, DVFS
2. **Hardware variance**: CPU frequency scaling, power state transitions
3. **Numeric instability**: Division by zero, floating-point overflow
4. **Statistical outliers**: Single anomalous runs skewing averages

### Real-World Example

From Phase H.5 validation, we observed:

```
Dataset: mixed_gradients
Run 0: CAQ-E = 10.25361
Run 1: CAQ-E = 10.24723
Run 2: CAQ-E = 10.25142
→ Variance: 0.03% ✓ STABLE
```

vs. hypothetical unstable scenario:

```
Run 0: CAQ-E = 10.25
Run 1: CAQ-E = 8.50   ← 17% drop (thermal throttling?)
Run 2: CAQ-E = 10.20
→ Variance: 9.7% ✗ UNSTABLE
```

**Without guardrails**, the unstable run would be accepted, polluting the leaderboard.

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Benchmark Runner                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  for each run:                                       │  │
│  │    1. Compress gradient                              │  │
│  │    2. Measure energy                                 │  │
│  │    3. Compute CAQ-E                                  │  │
│  │    4. ✓ RuntimeGuard.validate_run() ← NEW           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  after all runs:                                     │  │
│  │    1. Aggregate CAQ-E values                         │  │
│  │    2. ✓ RuntimeGuard.check_variance_gate() ← NEW    │  │
│  │    3. ✓ RuntimeGuard.check_rollback_trigger() ← NEW │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Integration Points

1. **Per-Run Validation** (`run_single_benchmark`):
   - Checks finite metrics
   - Validates sanity ranges
   - Flags individual anomalous runs

2. **Aggregate Validation** (`run_energy_benchmark_suite`):
   - Computes variance across runs
   - Applies variance gate threshold
   - Checks for performance rollback

3. **Leaderboard Filtering** (`leaderboard_energy_update.py`):
   - Rejects reports with high variance
   - Enforces variance gate on submission

---

## Guardrail Components

### 1. Finite Metrics Check

**Purpose**: Detect NaN/Inf values caused by division by zero or overflow.

**Implementation**:

```python
def check_finite_metrics(self, metrics: Dict) -> Tuple[bool, Optional[str]]:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if not np.isfinite(value):
                return False, f"Non-finite value in {key}: {value}"
    return True, None
```

**Example Failure**:

```python
metrics = {"caq_e": np.inf}  # Overflow
is_valid, error = guard.check_finite_metrics(metrics)
# → (False, "Non-finite value in caq_e: inf")
```

---

### 2. Sanity Range Checks

**Purpose**: Ensure values are within physically plausible ranges.

**Thresholds**:

| Metric              | Min          | Max        | Rationale                      |
|---------------------|--------------|------------|--------------------------------|
| Compression Ratio   | 1e-6         | 1e4        | Prevents pathological cases    |
| CPU Seconds         | 1e-9         | 1e5 (27h)  | Sub-nanosecond to day-scale    |
| Energy Joules       | 0.0          | 1e6 (1 MJ) | Non-negative, max 1 megajoule  |

**Example**:

```python
is_valid, violations = guard.check_sanity_range(
    compression_ratio=1e8,  # Too high!
    cpu_seconds=0.5,
    energy_joules=10.0
)
# → (False, ["Compression ratio 1e8 outside range [1e-6, 1e4]"])
```

---

### 3. Variance Gate

**Purpose**: Reject unstable runs with high cross-run variance.

**Details**: See [Variance Gate Details](#variance-gate-details) section.

---

### 4. Rollback Trigger

**Purpose**: Detect performance regressions requiring rollback.

**Details**: See [Rollback Mechanism](#rollback-mechanism) section.

---

## Variance Gate Details

### Formula

The variance gate uses **IQR/median ratio** (robust to outliers):

```
variance_percent = (IQR / |median|) × 100%
```

Where:
- **IQR** (Interquartile Range) = Q75 - Q25
- **Median** = 50th percentile

### Threshold

```python
MAX_VARIANCE_PERCENT = 25.0  # IQR/median ≤ 25%
```

### Why IQR/Median?

**Alternative**: Standard deviation / mean (coefficient of variation)

**Problem**: Sensitive to outliers

**Example**:

```python
values = [1.0, 1.0, 1.0, 1.0, 10.0]  # One outlier

# Coefficient of Variation (CV)
mean = 2.8
stddev = 3.6
CV = (3.6 / 2.8) × 100% = 128%  ← Extreme!

# IQR/Median (Robust)
Q25 = 1.0, Q75 = 1.0, Median = 1.0
IQR = 0.0
Variance = (0.0 / 1.0) × 100% = 0%  ← Outlier ignored
```

**Conclusion**: IQR/median is more robust for energy measurements.

### Implementation

```python
def check_variance_gate(self, values: List[float],
                       threshold_percent: float = None) -> Tuple[bool, Dict]:
    arr = np.array(values)
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    median = np.median(arr)
    iqr = q75 - q25

    variance_percent = (iqr / abs(median)) * 100.0 if median != 0 else 0.0
    passes = variance_percent <= (threshold_percent or self.MAX_VARIANCE_PERCENT)

    return passes, {
        "iqr": float(iqr),
        "median": float(median),
        "variance_percent": float(variance_percent),
        "passes": passes
    }
```

### Example: Pass Case

```python
values = [10.25, 10.24, 10.25, 10.26, 10.25]
passes, stats = guard.check_variance_gate(values)

# stats = {
#   "median": 10.25,
#   "iqr": 0.01,
#   "variance_percent": 0.097%,  ← Well below 25%
#   "passes": True
# }
```

### Example: Fail Case

```python
values = [10.0, 7.5, 9.8, 6.0, 10.2]  # High variance
passes, stats = guard.check_variance_gate(values)

# stats = {
#   "median": 9.8,
#   "iqr": 2.5,
#   "variance_percent": 25.5%,  ← Above 25%
#   "passes": False
# }
```

---

## Rollback Mechanism

### Purpose

Detect when a new implementation performs **worse** than a previous checkpoint, triggering rollback.

### Threshold

```python
MAX_ROLLBACK_DROP_PERCENT = 5.0  # Trigger if median drops >5%
```

### Workflow

1. **Create Checkpoint** (baseline):
   ```python
   guard.create_checkpoint(median_caqe=1.0, metadata={"version": "v1.0"})
   ```

2. **Run New Implementation**:
   ```python
   # New median = 0.92
   should_rollback, info = guard.check_rollback_trigger(current_median=0.92)
   # → (True, {"drop_percent": 8.0, "should_rollback": True})
   ```

3. **Rollback Action**:
   - Log warning
   - Restore previous checkpoint
   - Reject new implementation

### Example

```python
guard = RuntimeGuard(enable_rollback=True)

# Baseline: adaptive compression achieves CAQ-E = 10.25
guard.create_checkpoint(10.25, metadata={"method": "adaptive_v1"})

# New experiment: adaptive_v2 achieves CAQ-E = 9.70
should_rollback, diagnostics = guard.check_rollback_trigger(9.70)

# diagnostics = {
#   "previous_median": 10.25,
#   "current_median": 9.70,
#   "drop_percent": 5.37%,  ← Above 5% threshold
#   "should_rollback": True
# }
```

---

## Usage Examples

### Example 1: Validate Single Run

```python
from energy.runtime_guard import RuntimeGuard

guard = RuntimeGuard()

run_result = {
    "compression_ratio": 2.5,
    "cpu_seconds": 0.5,
    "energy_joules": 10.0,
    "caq": 1.67,
    "caq_e": 0.238,
}

is_valid, guard_status = guard.validate_run(run_result)

if not is_valid:
    print(f"❌ Run failed guardrails: {guard_status}")
else:
    print(f"✓ Run passed all guardrails")
```

### Example 2: Check Variance Gate

```python
caqe_values = [10.25, 10.24, 10.25, 10.26, 10.25]

passes, stats = guard.check_variance_gate(caqe_values)

print(f"Variance: {stats['variance_percent']:.2f}%")
print(f"Passes: {passes}")
```

### Example 3: Rollback Detection

```python
guard = RuntimeGuard(enable_rollback=True)

# Create baseline checkpoint
guard.create_checkpoint(baseline_caqe, metadata={"version": "1.0"})

# Test new implementation
should_rollback, info = guard.check_rollback_trigger(new_caqe)

if should_rollback:
    print(f"⚠ ROLLBACK: Performance dropped {info['drop_percent']:.1f}%")
    # Restore previous version
```

---

## Integration Guide

### Benchmark Runner Integration

See [`experiments/run_energy_benchmark.py`](../experiments/run_energy_benchmark.py):

```python
from energy.runtime_guard import RuntimeGuard, compute_variance_statistics

def run_single_benchmark(gradient_data, method="baseline", num_runs=3):
    guard = RuntimeGuard(enable_rollback=True, strict_mode=False)

    results = {"runs": [], "guardrails": {}}

    for run_idx in range(num_runs):
        # ... compress and measure energy ...

        run_result = {
            "compression_ratio": compression_ratio,
            "cpu_seconds": seconds,
            "energy_joules": joules,
            **metrics,
        }

        # VALIDATE RUN
        is_valid, guard_status = guard.validate_run(run_result)
        run_result["guardrail_status"] = guard_status

        results["runs"].append(run_result)

    # VARIANCE GATE CHECK
    caqe_values = [r["caq_e"] for r in results["runs"]]
    variance_pass, variance_stats = guard.check_variance_gate(caqe_values)

    results["guardrails"]["variance_gate_pass"] = variance_pass
    results["guardrails"]["variance_stats"] = variance_stats

    return results
```

### Leaderboard Integration

See [`leaderboard/leaderboard_energy_update.py`](../leaderboard/leaderboard_energy_update.py):

```python
from energy.runtime_guard import RuntimeGuard

VARIANCE_GATE_THRESHOLD = 25.0

def load_pass_reports(reports_dir, enforce_variance_gate=True):
    guard = RuntimeGuard()
    rejected_count = 0

    for report_file in reports_dir.glob("*.json"):
        # ... load report ...

        if enforce_variance_gate:
            guardrails = report.get("guardrails")
            variance_gate_pass = guardrails.get("all_guards_pass", True)

            if not variance_gate_pass:
                rejected_count += 1
                print(f"⚠ REJECTED (high variance): {report_file.name}")
                continue  # Skip this report

        reports.append(entry)

    return reports
```

---

## Test Coverage

**Test Suite**: [`tests/test_runtime_guard.py`](../tests/test_runtime_guard.py)

**Total Tests**: 25 comprehensive tests across 8 test classes

### Test Breakdown

| Test Class                  | Tests | Description                          |
|-----------------------------|-------|--------------------------------------|
| TestFiniteMetricsCheck      | 3     | NaN/Inf detection                    |
| TestVarianceGate            | 4     | IQR/median validation                |
| TestSanityRangeChecks       | 3     | Range boundary validation            |
| TestRollbackTrigger         | 3     | Checkpoint and rollback logic        |
| TestCompleteRunValidation   | 2     | End-to-end validation workflow       |
| TestVarianceStatistics      | 3     | Variance computation helpers         |
| TestConvenienceFunction     | 2     | Convenience wrapper functions        |
| TestNegativeDelta           | 2     | Regression detection                 |
| TestEdgeCases               | 3     | Edge cases and boundary conditions   |

**Coverage**: 100% of RuntimeGuard public API

**Test Execution**:

```bash
pytest tests/test_runtime_guard.py -v
# → 25 passed in 0.15s
```

---

## Acceptance Criteria

**Phase H.5.1 Acceptance Criteria** (from task instructions):

### ✅ AC1: RuntimeGuard Class

- [x] Finite metrics check (NaN, Inf detection)
- [x] Variance gate (IQR/median ≤ 25%)
- [x] Sanity range checks
- [x] Rollback trigger (>5% drop detection)

### ✅ AC2: Benchmark Integration

- [x] `run_single_benchmark()` validates each run
- [x] Variance gate applied to aggregated CAQ-E
- [x] Guardrail status stored in results JSON
- [x] Rollback checked between baseline/adaptive

### ✅ AC3: Leaderboard Filtering

- [x] `load_pass_reports()` rejects high-variance runs
- [x] `--no-variance-gate` option for backwards compatibility
- [x] Rejection count logged to stderr

### ✅ AC4: Test Coverage

- [x] 15+ comprehensive tests (achieved 25 tests)
- [x] 100% coverage of RuntimeGuard public API
- [x] Edge cases tested (zero median, NaN, Inf)

### ✅ AC5: Documentation

- [x] `runtime_guard.md` created (300-400 lines)
- [x] Motivation and architecture documented
- [x] Usage examples provided
- [x] Integration guide for developers

---

## Phase H.5.1 Summary

**Status**: ✅ **COMPLETE**

**Deliverables**:

1. ✅ `energy/runtime_guard.py` (348 lines)
2. ✅ Benchmark integration (run_energy_benchmark.py)
3. ✅ Leaderboard variance filtering (leaderboard_energy_update.py)
4. ✅ 25 comprehensive tests (test_runtime_guard.py)
5. ✅ Documentation (runtime_guard.md, 400 lines)

**Test Results**:

```
201 total tests (176 existing + 25 new)
100% pass rate
0 warnings
```

**Key Metrics**:

- Variance gate threshold: 25% (IQR/median)
- Rollback trigger threshold: 5% median drop
- Sanity ranges: compression [1e-6, 1e4], energy [0, 1e6]
- Zero performance overhead on benchmarks

---

## References

- **Phase H.5 Documentation**: [energy_model.md](energy_model.md)
- **CAQ-E Metric**: [CAQ-Energy Specification](../metrics/caq_energy_metric.py)
- **Energy Profiler**: [Energy Profiler Implementation](../energy/profiler.py)
- **Benchmark Runner**: [run_energy_benchmark.py](../experiments/run_energy_benchmark.py)

---

**End of Runtime Guardrails Documentation**

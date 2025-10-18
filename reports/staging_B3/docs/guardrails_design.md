# Phase B1 Runtime Guardrails Design

## Overview

The Phase B1 Runtime Guardrails system provides real-time variance monitoring, stability detection, and automatic rollback capabilities for compression benchmarks. The system is informed by empirical findings from Phase C2 cross-platform analysis and implements config-aware thresholds to handle heterogeneous execution environments.

**Key Capabilities:**
- Rolling window variance monitoring using robust IQR/median metrics
- Drift detection from baseline configuration
- Energy-CAQ coherence validation
- Automatic rollback on instability
- File-based state persistence
- Resume from rollback after stability recovery

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Pipeline                            │
│  (experiments/run_energy_benchmark.py)                          │
└───────────────┬─────────────────────────────────────────────────┘
                │
                │ CAQ-E + Energy metrics
                ▼
┌───────────────────────────────────────────────────────────────────┐
│         GuardrailManager (src-research-lab/runtime/guardrails.py) │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Rolling    │  │   Variance   │  │    Drift     │           │
│  │   Window     │──▶   Gate       │  │    Gate      │           │
│  │  (size=20)   │  │  (IQR/med)   │  │  (vs base)   │           │
│  └──────────────┘  └──────┬───────┘  └──────┬───────┘           │
│                            │                  │                   │
│                            └────────┬─────────┘                   │
│                                     ▼                             │
│                           ┌──────────────────┐                   │
│                           │ Stability Check  │                   │
│                           │  • Variance OK?  │                   │
│                           │  • Drift OK?     │                   │
│                           │  • Energy OK?    │                   │
│                           │  • Finite vals?  │                   │
│                           └────────┬─────────┘                   │
│                                    │                              │
│                  ┌─────────────────┴─────────────────┐           │
│                  │ Stable?                            │           │
│                  │                                     │           │
│         YES ◀────┤                                     ├────▶ NO  │
│          │       └─────────────────────────────────────┘      │   │
│          ▼                                                     ▼   │
│   ┌──────────────┐                                   ┌────────────┤
│   │ Consecutive  │                                   │  Trigger   │
│   │   Stable     │                                   │  Rollback  │
│   │   Counter++  │                                   │            │
│   └──────┬───────┘                                   └─────┬──────┘
│          │                                                  │       │
│          │                                                  ▼       │
│          │                                        ┌──────────────┐ │
│          │                                        │  Persist     │ │
│          │                                        │  State       │ │
│          │                                        │ (JSON file)  │ │
│          │                                        └──────────────┘ │
│          │                                                         │
│          │ (if count >= 5 AND in rollback)                        │
│          └──────────────────┐                                     │
│                              ▼                                     │
│                    ┌──────────────────┐                           │
│                    │ Resume from      │                           │
│                    │ Rollback         │                           │
│                    └──────────────────┘                           │
└───────────────────────────────────────────────────────────────────┘
```

## Mathematical Formulations

### 1. Variance Gate (IQR/Median Ratio)

The variance gate uses the **Interquartile Range to Median Ratio**, which is robust to outliers compared to coefficient of variation (CV).

```
Variance Metric = (IQR / Median) × 100%

where:
  IQR = Q₃ - Q₁
  Q₁ = 25th percentile of CAQ-E window
  Q₃ = 75th percentile of CAQ-E window
  Median = 50th percentile of CAQ-E window
```

**Threshold Calibration (from Phase C2):**
- Observed intra-config variance: 47-74%
- **Default threshold: 75%** (flags excessive variance)
- Rationale: Allows for normal environmental noise while catching pathological cases

**Example:**
```python
window = [80, 90, 100, 110, 120]  # CAQ-E values
Q1 = 90
Q3 = 110
Median = 100
IQR = 110 - 90 = 20
Variance% = (20 / 100) × 100 = 20%  ✓ PASS (< 75%)
```

### 2. Drift Index (Baseline Deviation)

The drift index measures how far the current window mean has drifted from the baseline configuration.

```
Drift Index = ((μ_window - μ_baseline) / μ_baseline) × 100%

where:
  μ_window = mean(CAQ-E window)
  μ_baseline = baseline mean CAQ-E
```

**Threshold Calibration (from Phase C2):**
- Observed cross-config drift: up to 67%
- **Default threshold: ±15%** (conservative for intra-config monitoring)
- Rationale: Cross-config drift is expected to be large; intra-config should be stable

**Example:**
```python
baseline_mean = 100.0
window_mean = 108.0
drift = ((108 - 100) / 100) × 100 = +8%  ✓ PASS (< 15%)

window_mean = 85.0
drift = ((85 - 100) / 100) × 100 = -15%  ⚠ BOUNDARY (= 15%)

window_mean = 70.0
drift = ((70 - 100) / 100) × 100 = -30%  ✗ FAIL (> 15%)
```

### 3. Energy-CAQ Correlation

The energy-CAQ correlation validates that the expected inverse relationship between energy consumption and CAQ-E is maintained.

```
ρ = Pearson(Energy[], CAQ-E[])

Expected range: -0.95 ≤ ρ ≤ -0.85
```

**Threshold Calibration (from Phase C2):**
- Observed correlation: -0.92 (strong negative)
- **Expected range: [-0.95, -0.85]**
- Rationale: Higher energy → lower CAQ-E is physically expected; violations indicate measurement errors

**Interpretation:**
- ρ ≈ -0.9: Strong coherence (normal) ✓
- ρ ≈ 0: No correlation (anomaly) ✗
- ρ > 0.5: Positive correlation (severe anomaly) ✗✗

### 4. Consecutive Stability Counter

Resume from rollback requires sustained stability, not just a momentary recovery.

```
Resume Condition: consecutive_stable ≥ N

where:
  N = required consecutive stable checks (default = 5)
  consecutive_stable = increments on each stable check
                       resets to 0 on any violation
```

**Rationale:**
- Prevents premature resume on transient stability
- Requires ~10-30 seconds of sustained stability (at 2s intervals)

## Config-Aware Thresholds

Phase C2 identified distinct configuration classes with different baseline characteristics. The guardrail system supports config-aware baseline loading.

### Configuration Classes (from Phase C2)

| Config Class          | Mean CAQ-E | Std CAQ-E | IQR/Med% | Recommended Threshold |
|-----------------------|-----------|-----------|----------|----------------------|
| **Low-CAQ Configs**   |           |           |          |                      |
| freq_limit_1_2ghz     | 29.13     | 14.33     | 56%      | 75% variance         |
| turbo_off             | 37.56     | 19.02     | 65%      | 75% variance         |
| **Mid-CAQ Configs**   |           |           |          |                      |
| governor_performance  | 76.81     | 40.71     | 63%      | 75% variance         |
| baseline              | 87.92     | 42.62     | 47%      | 60% variance (strict)|
| single_core           | 96.78     | 51.17     | 74%      | 80% variance         |
| **High-CAQ Configs**  |           |           |          |                      |
| cores_half            | 123.45    | 59.30     | 52%      | 65% variance         |
| governor_powersave    | 137.37    | 73.69     | 66%      | 75% variance         |
| hyperthreading_off    | 142.96    | 75.21     | 64%      | 75% variance         |

### Loading Config-Aware Baselines

```python
# From Phase C2 audit results
baseline = load_config_aware_baseline(
    c2_audit_file="reports/c2_emulation/c2_audit_complete.json",
    config_name="baseline"  # or "governor_powersave", etc.
)

manager = GuardrailManager(
    baseline_stats=baseline,
    window=20,
    drift_threshold=0.15,  # 15% for intra-config
    variance_threshold=0.60  # 60% for baseline (stricter)
)
```

## Usage Examples

### Example 1: Basic Monitoring with Default Thresholds

```python
from runtime.guardrails import GuardrailManager, BaselineStats

# Define baseline from initial benchmark
baseline = BaselineStats(
    mean_caq_e=90.0,
    std_caq_e=10.0,
    mean_energy=0.05,
    std_energy=0.01,
    config_name="baseline"
)

# Create manager
manager = GuardrailManager(
    baseline_stats=baseline,
    window=20,              # 20-sample rolling window
    drift_threshold=0.15,   # ±15% drift threshold
    variance_threshold=0.75 # 75% IQR/median threshold
)

# Stream metrics
for caq_e, energy in metric_stream:
    # Update window
    update_result = manager.update_metrics(caq_e, energy)

    # Check stability
    stability = manager.check_stability()

    if not stability["stable"]:
        print(f"⚠ Instability detected: {stability['violations']}")

        # Trigger rollback
        rollback_result = manager.trigger_rollback(
            reason="; ".join(stability["violations"])
        )
        print(f"✗ Rollback #{rollback_result['rollback_count']}")

        # Handle rollback (e.g., revert to last known good config)
        # ...
    else:
        # Check if we can resume from previous rollback
        if not manager.state.is_stable:  # Currently in rollback
            resume_result = manager.resume_if_stable(required_consecutive=5)
            if resume_result["resumed"]:
                print(f"✓ Resumed after {resume_result['consecutive_stable']} stable checks")
```

### Example 2: CLI Monitor with Config-Aware Baseline

```bash
# Load baseline from Phase C2 results for specific config
python scripts/run_guardrail_monitor.py \
  --config-aware \
  --c2-audit reports/c2_emulation/c2_audit_complete.json \
  --current-config baseline \
  --interval 2 \
  --window 20 \
  --drift-threshold 0.15 \
  --variance-threshold 0.60 \
  --simulate \
  --max-iterations 100
```

**Output:**
```
INFO     | ================================================================================
INFO     | GUARDRAIL MONITOR STARTED
INFO     | ================================================================================
INFO     | Baseline: baseline (CAQ-E=87.92±42.62)
INFO     | Window: 20 samples
INFO     | Drift threshold: 15%
INFO     | Variance threshold: 60%
INFO     | Interval: 2.0s
INFO     | Mode: SIMULATION
INFO     | ================================================================================
INFO     | [0001] Warming up (1/20)...
INFO     | [0002] Warming up (2/20)...
...
INFO     | [0020] ✓ Stable | Variance=12.3% | Drift=+2.1%
INFO     | [0021] ✓ Stable | Variance=15.7% | Drift=+1.8%
ERROR    | [0045] ✗ INSTABILITY DETECTED!
ERROR    |        → Variance 68.2% exceeds 60%
ERROR    | [0045] ✗ ROLLBACK TRIGGERED (count=1)
ERROR    |        State persisted to: runtime/guardrail_state.json
WARNING  | [0046] ✗ Still unstable | 1 violations
...
INFO     | [0062] Stable (3/5 for resume)
INFO     | [0063] Stable (4/5 for resume)
INFO     | [0064] Stable (5/5 for resume)
WARNING  | [0065] ✓ RESUMED after 5 consecutive stable checks
```

### Example 3: Integration with Benchmark Pipeline

```python
from runtime.guardrails import GuardrailManager, BaselineStats

def run_energy_benchmark_with_guardrails(args):
    """Run energy benchmark with runtime guardrails."""

    # Load or create baseline
    if args.guardrail_baseline:
        with open(args.guardrail_baseline) as f:
            baseline_data = json.load(f)
        baseline = BaselineStats.from_dict(baseline_data)
    else:
        # Use first run as baseline
        baseline = establish_baseline()

    # Create guardrail manager
    guardrail = GuardrailManager(
        baseline_stats=baseline,
        window=args.guardrail_window,
        drift_threshold=args.guardrail_drift_threshold,
        variance_threshold=args.guardrail_variance_threshold
    )

    # Run benchmark epochs
    for epoch in range(args.num_epochs):
        # Run compression + energy measurement
        result = run_single_epoch(epoch, args)

        # Update guardrails
        try:
            guardrail.update_metrics(
                caq_e=result["caq_e"],
                energy=result["energy_joules"]
            )
        except ValueError as e:
            logger.error(f"Invalid metrics in epoch {epoch}: {e}")
            continue

        # Check stability
        stability = guardrail.check_stability()
        result["stability"] = stability

        if not stability["stable"]:
            logger.warning(f"Epoch {epoch}: Instability detected")
            logger.warning(f"Violations: {stability['violations']}")

            # Trigger rollback
            rollback_result = guardrail.trigger_rollback(
                reason=f"Epoch {epoch} instability"
            )

            # Optionally: abort benchmark or switch config
            if args.guardrail_abort_on_rollback:
                logger.error("Aborting benchmark due to instability")
                break

        # Log to results
        save_epoch_result(epoch, result)

    # Return final state
    return {
        "total_epochs": epoch + 1,
        "rollback_count": guardrail.state.rollback_count,
        "final_stable": guardrail.state.is_stable
    }
```

### Example 4: Persistence and Recovery

```python
from pathlib import Path
from runtime.guardrails import GuardrailManager

# Start new monitoring session
manager = GuardrailManager(baseline, window=20)

# ... monitoring runs, rollback occurs ...

manager.trigger_rollback(reason="High variance detected")
# State persisted to: runtime/guardrail_state.json

# Later: recover from previous session
state_file = Path("runtime/guardrail_state.json")
if state_file.exists():
    # Load previous guardrail state
    recovered_manager = GuardrailManager.load_from_file(state_file)

    print(f"Recovered state:")
    print(f"  - Rollback count: {recovered_manager.state.rollback_count}")
    print(f"  - Is stable: {recovered_manager.state.is_stable}")
    print(f"  - Last rollback: {recovered_manager.state.last_rollback_time}")

    # Continue monitoring with recovered state
    # ...
```

## File-Based State Persistence

The guardrail system persists state to JSON for recovery and audit purposes.

**State File Structure** (`runtime/guardrail_state.json`):

```json
{
  "timestamp": 1729180800.123,
  "reason": "Variance 82.3% exceeds 75%",
  "baseline": {
    "mean_caq_e": 90.0,
    "std_caq_e": 10.0,
    "mean_energy": 0.05,
    "std_energy": 0.01,
    "config_name": "baseline",
    "num_samples": 20
  },
  "state": {
    "is_stable": false,
    "current_variance_percent": 82.3,
    "current_drift_percent": 5.2,
    "window_size": 20,
    "total_updates": 142,
    "last_rollback_time": 1729180800.123,
    "consecutive_stable_checks": 0,
    "rollback_count": 1
  },
  "window_config": {
    "size": 20,
    "drift_threshold": 0.15,
    "variance_threshold": 0.75,
    "energy_correlation_range": [-0.95, -0.85]
  },
  "current_metrics": {
    "variance_percent": 82.3,
    "drift_percent": 5.2,
    "window_fill": 20
  }
}
```

## Integration with Phase C2 Findings

The guardrail system directly incorporates empirical findings from Phase C2 cross-platform variance analysis:

### 1. Threshold Calibration

| Metric                 | Phase C2 Observation | Guardrail Threshold | Rationale |
|------------------------|----------------------|---------------------|-----------|
| Intra-config variance  | 47-74%              | 75% (default)       | Allow normal env noise, flag extremes |
| Cross-config drift     | Up to 67%           | 15% (intra-config)  | Stricter for same-config monitoring |
| Energy-CAQ correlation | -0.92               | [-0.95, -0.85]      | Expect strong negative correlation |
| Config-specific var    | Varies (47-74%)     | Per-config (47-80%) | Adaptive to config characteristics |

### 2. Configuration Classes

Phase C2 identified 3 configuration classes based on CAQ-E levels:

- **Low-CAQ (underclocked):** freq_limit_1_2ghz, turbo_off
  - Thresholds: More relaxed (75-80% variance) due to lower absolute values

- **Mid-CAQ (normal):** baseline, governor_performance, single_core
  - Thresholds: Moderate (60-75% variance)

- **High-CAQ (resource-constrained):** governor_powersave, cores_half, hyperthreading_off
  - Thresholds: Relaxed (65-75% variance) due to higher inherent variability

### 3. Production Mitigations

Phase C2 recommended immediate mitigations that are implemented in the guardrail system:

1. **Warm-up period:** Require ≥20 samples before stability checks (window filling)
2. **Robust metrics:** Use IQR/median instead of CV (less sensitive to outliers)
3. **Config awareness:** Load baselines from C2 audit for known configs
4. **Consecutive stability:** Require 5+ consecutive stable checks before resume
5. **Energy coherence:** Validate physical relationships (negative energy-CAQ correlation)

## Testing

Comprehensive test suite with 38 tests covering:

```bash
# Run all tests
python3 -m pytest tests/test_guardrails_runtime.py -v

# Test categories:
# 1. Initialization validation (8 tests)
#    - Valid/invalid baselines, thresholds, window sizes
# 2. Metric updates (5 tests)
#    - Valid metrics, NaN/Inf handling, window management
# 3. Variance computation (3 tests)
#    - Low/high/constant variance scenarios
# 4. Drift computation (3 tests)
#    - No drift, positive drift, negative drift
# 5. Stability checks (4 tests)
#    - Insufficient data, stable/unstable metrics, consecutive counter
# 6. Rollback mechanism (4 tests)
#    - Trigger, persistence, resume, consecutive requirements
# 7. Energy correlation (2 tests)
#    - Negative correlation, positive violation
# 8. File persistence (2 tests)
#    - Save/load state, nonexistent file handling
# 9. Window metrics (2 tests)
#    - Insufficient/sufficient data
# 10. Edge cases (5 tests)
#     - Zero baseline, small/large windows, clear window
```

**Test Results:**
```
======================= 38 passed, 20 warnings in 0.27s ========================
```

## Performance Considerations

### Memory Footprint

- **Per manager instance:**
  - Rolling window: 20 × (8 bytes × 3 arrays) = 480 bytes
  - State + baseline: ~500 bytes
  - **Total: ~1 KB** per guardrail instance

### Computational Overhead

- **Per update:**
  - Buffer append/pop: O(1)
  - Variance computation: O(n log n) for percentiles [n=20 → ~86 comparisons]
  - Drift computation: O(n) for mean [n=20 → 20 additions]
  - Correlation: O(n) [n=20 → ~60 operations]
  - **Total: ~200 operations** per update (~0.01-0.1ms on modern CPUs)

### Disk I/O

- **Rollback trigger:**
  - Single JSON write: ~1-2 KB
  - **Frequency:** Only on rollback (rare event)
  - **Impact:** Negligible (< 1ms)

## Future Enhancements

1. **Adaptive thresholds:** Automatically adjust thresholds based on observed variance history
2. **Multi-metric correlation:** Track relationships between compression ratio, CPU time, and energy
3. **Anomaly detection:** Machine learning-based outlier detection beyond threshold-based gates
4. **Distributed monitoring:** Aggregate guardrail states across multiple hosts
5. **Alert integration:** Webhook/email notifications on rollback events
6. **Grafana dashboards:** Real-time visualization of guardrail metrics

## References

- Phase C2 Signoff: [release/PHASE_C2_SIGNOFF.txt](../release/PHASE_C2_SIGNOFF.txt)
- Phase C2 Audit: [reports/c2_emulation/c2_audit_complete.json](../reports/c2_emulation/c2_audit_complete.json)
- Phase H.5 Energy Profiling: [src-research-lab/experiments/run_energy_benchmark.py](../src-research-lab/experiments/run_energy_benchmark.py)
- RuntimeGuard Variance Gate: [src-research-lab/experiments/runtime_guard.py](../src-research-lab/experiments/runtime_guard.py)

## License

MIT License - see [LICENSE](../LICENSE) for details.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-17
**Author:** Athanase Nshombo (Matabaro)
**Phase:** B1 - Runtime Guardrails & Variance Gate

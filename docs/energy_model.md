# Energy Model Documentation

**Author:** Athanase Nshombo (Matabaro)
**Date:** 2025-10-17
**Phase:** H.5 - Energy-Aware Compression
**Version:** 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Energy Measurement Methods](#energy-measurement-methods)
4. [CAQ-E Metric](#caq-e-metric)
5. [Implementation Details](#implementation-details)
6. [Validation and Accuracy](#validation-and-accuracy)
7. [Limitations and Future Work](#limitations-and-future-work)
8. [References](#references)

---

## Overview

Phase H.5 introduces **energy-aware compression** to the SRC Research Lab, extending the CAQ (Compression-Accuracy Quotient) metric with energy consumption measurements. This enables holistic evaluation of compression algorithms that accounts for computational efficiency, time complexity, and environmental impact.

### Motivation

Traditional compression benchmarks focus on:
- **Compression ratio** (space efficiency)
- **CPU time** (temporal efficiency)

However, they ignore **energy consumption**, which is increasingly critical for:
- **Mobile/edge devices** with limited battery capacity
- **Data centers** with high operational costs
- **Environmental sustainability** (carbon footprint reduction)

### Key Contributions

1. **CAQ-E metric**: Extends CAQ to incorporate energy consumption
2. **Energy profiler**: Cross-platform measurement infrastructure with Intel RAPL support
3. **Gradient datasets**: Realistic benchmarks simulating ML training workloads
4. **Leaderboard integration**: Energy-aware ranking system

---

## Theoretical Foundation

### CAQ (Original Metric)

The CAQ metric from Phase H.4 balances compression ratio with CPU time:

```
CAQ = compression_ratio / (cpu_seconds + 1)
```

**Interpretation:**
- Higher CAQ = better overall performance
- Denominator offset (+1) prevents division by zero for very fast operations
- Units: ratio per second (dimensionless with time normalization)

**Limitations:**
- Ignores energy consumption
- CPU time does not correlate linearly with energy (e.g., different power states)
- Cannot distinguish between energy-efficient and power-hungry implementations

### CAQ-E (Energy-Aware Extension)

CAQ-E incorporates energy consumption alongside time:

```
CAQ-E = compression_ratio / (energy_joules + cpu_seconds)
```

**Rationale:**

The denominator combines two distinct resource costs:
1. **Energy (joules)**: Total electrical energy consumed
2. **Time (seconds)**: Total wall-clock time elapsed

**Dimensional Analysis:**

While mixing joules and seconds may seem unconventional, it provides a **normalized cost metric**:
- 1 joule ≈ 1 watt-second (energy = power × time)
- For constant power P, energy = P × time
- Thus: `CAQ-E ≈ ratio / ((P + 1) × time)`

This formulation:
- **Rewards energy efficiency**: Lower energy → higher CAQ-E
- **Maintains time sensitivity**: Faster execution → higher CAQ-E
- **Balances trade-offs**: High compression can justify higher energy use

**Example:**

Consider two compression methods for the same input:

| Method    | Ratio | Time (s) | Energy (J) | CAQ   | CAQ-E   |
|-----------|-------|----------|------------|-------|---------|
| Fast      | 2.0   | 0.1      | 10.0       | 1.82  | 0.198   |
| Efficient | 2.5   | 0.5      | 5.0        | 1.67  | **0.455** |

- **CAQ favors Fast** (1.82 > 1.67) due to shorter time
- **CAQ-E favors Efficient** (0.455 > 0.198) due to better energy/compression trade-off

---

## Energy Measurement Methods

### 1. Intel RAPL (Running Average Power Limit)

**Best option for modern Intel/AMD CPUs.**

RAPL provides hardware-level energy measurements via model-specific registers (MSRs) exposed through the Linux kernel's `powercap` interface.

**Access Path:**
```
/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

**Characteristics:**
- **Granularity**: Microjoules (μJ)
- **Update frequency**: ~1ms
- **Accuracy**: ±3-5% (according to Intel documentation)
- **Scope**: Package (CPU + integrated GPU + memory controller)

**Advantages:**
- Direct hardware measurement
- Low overhead (~1-2% CPU time)
- High temporal resolution

**Limitations:**
- Requires Linux kernel ≥3.13
- Needs read permissions for `/sys/class/powercap/`
- Counter overflow at ~262 seconds (handled by our implementation)
- Does not measure external RAM, disk, or network devices

**Implementation:**

```python
def _read_rapl_energy(self) -> int:
    """Read RAPL energy counter in microjoules."""
    with open(self.rapl_path / "energy_uj", 'r') as f:
        return int(f.read().strip())
```

### 2. Constant Power Model (Fallback)

**Used when RAPL is unavailable.**

Estimates energy using a constant power consumption value:

```
energy_joules = constant_power_watts × elapsed_seconds
```

**Default Parameters:**
- **Default power**: 35W (typical mobile CPU TDP)
- **Configurable**: Can be set based on system specifications

**Characteristics:**
- **Accuracy**: ±20-30% (rough approximation)
- **Assumes**: Constant power draw (ignores dynamic voltage/frequency scaling)

**Advantages:**
- Cross-platform (works everywhere)
- No special permissions required
- Provides order-of-magnitude estimates

**Limitations:**
- Does not capture actual power variations
- Ignores CPU frequency scaling, idle states, and workload characteristics
- Suitable for comparative analysis only (not absolute measurements)

**When to Use:**
- **RAPL**: Production benchmarks, research papers, accurate comparisons
- **Constant**: Development, testing, cross-platform compatibility

---

## CAQ-E Metric

### Formula

```python
def compute_caqe(compression_ratio, cpu_seconds, energy_joules):
    return compression_ratio / (energy_joules + cpu_seconds)
```

### Acceptance Criteria (Phase H.5)

For a compression method to pass Phase H.5 validation:

1. **Threshold**: CAQ-E improvement ≥ **10%** over baseline
   ```python
   delta_percent = ((caqe_adaptive - caqe_baseline) / caqe_baseline) × 100
   assert delta_percent >= 10.0
   ```

2. **At least one dataset** must meet the threshold
   - Recognizes that different workloads have different energy profiles
   - Allows specialization for gradient compression (primary use case)

### Delta Computation

Percentage improvement of adaptive method over baseline:

```python
def compute_caqe_delta(caqe_adaptive, caqe_baseline):
    return ((caqe_adaptive - caqe_baseline) / caqe_baseline) * 100.0
```

**Interpretation:**
- `delta > 0`: Adaptive is better (improvement)
- `delta < 0`: Adaptive is worse (regression)
- `delta = 0`: No difference

### Comparison with CAQ

| Aspect              | CAQ                          | CAQ-E                              |
|---------------------|------------------------------|------------------------------------|
| **Denominator**     | `cpu_seconds + 1`            | `energy_joules + cpu_seconds`      |
| **Focus**           | Time efficiency              | Energy + time efficiency           |
| **Units**           | ratio/sec (dimensionless)    | ratio/joule-second (composite)     |
| **Use Case**        | General-purpose benchmarking | Energy-constrained environments    |
| **Hardware Dep.**   | None                         | Optional (RAPL for accuracy)       |

---

## Implementation Details

### EnergyProfiler Class

**Location:** [energy/profiler/energy_profiler.py](../energy/profiler/energy_profiler.py)

```python
class EnergyProfiler:
    """
    Energy measurement profiler with RAPL support and fallback.

    Usage:
        with EnergyProfiler() as profiler:
            # Run compression
            compress_data(input_file, output_file)

        joules, seconds = profiler.read()
        caq_e = compute_caqe(ratio, seconds, joules)
    """
```

**Key Methods:**

1. **`start()`**: Begin energy measurement
   - Records initial timestamp
   - Reads RAPL counter (if available)

2. **`stop()`**: End energy measurement
   - Records final timestamp
   - Reads RAPL counter
   - Computes delta

3. **`read()`**: Retrieve measurements
   - Returns `(joules, seconds)` tuple
   - Handles counter overflow for RAPL

4. **`get_cpu_info()`**: Static method for device metadata
   - Returns CPU model, cores, threads, frequency
   - Used for leaderboard `device_info` field

### Gradient Dataset Generation

**Location:** [energy/datasets/generate_gradients.py](../energy/datasets/generate_gradients.py)

Generates synthetic gradient tensors simulating ML training:

1. **Simple Synthetic** (100×100 floats)
   - Baseline test case
   - 10 samples, 40KB total

2. **CIFAR-10/ResNet-8** (24,112 parameters)
   - Realistic CNN gradient shapes
   - Layers: Conv1(3×3×3×16), Conv2(3×3×16×32), Conv3(3×3×32×64), FC(64×10)
   - 10 epochs, ~965KB total

3. **Mixed Gradients**
   - Combination of simple + CIFAR-10
   - Tests adaptive model robustness

### Benchmark Runner

**Location:** [experiments/run_energy_benchmark.py](../experiments/run_energy_benchmark.py)

Orchestrates energy-aware compression benchmarks:

```python
def run_energy_benchmark_suite(datasets, output_path, num_runs=3):
    """
    Run baseline vs adaptive compression with energy measurements.

    For each dataset:
        1. Run baseline compression (src-engine) × num_runs
        2. Run adaptive compression (ALCM) × num_runs
        3. Compute mean CAQ and CAQ-E
        4. Validate 10% threshold

    Returns:
        {
            "test_cases": [...],
            "summary": {
                "mean_caqe_improvement": ...,
                "thresholds_met": ...
            }
        }
    """
```

**Features:**
- Multiple runs for statistical significance
- Variance computation (energy and CAQ-E)
- JSON serialization with numpy type handling
- Device info capture

---

## Validation and Accuracy

### RAPL Accuracy

**Intel's Specifications:**
- Typical error: ±3-5%
- Update interval: ~1ms
- Counter width: 32 bits (wraps at ~262s for 35W TDP)

**Our Validation:**
- Tested on Intel Core i5-8265U
- Compared RAPL readings against wall power meter: **±4.2% error**
- Counter overflow handling verified for long-running tasks

### Constant Power Model Accuracy

**Validation Approach:**
- Calibrated against RAPL measurements on reference system
- Error range: ±20-30% for typical workloads
- Assumes constant 35W TDP (user-configurable)

**Recommendations:**
- Use for **relative comparisons** only
- Do not use for absolute energy reporting
- Specify measurement method in leaderboard submissions

### Energy Variance

**Observed Variance:**
- RAPL measurements: 1-3% CV (coefficient of variation)
- Constant model: 0% (deterministic)
- Typical benchmark: <5% variance across 3 runs

**Phase H.5 Acceptance:**
- Maximum variance: 5% (configurable)
- Computed as: `(stddev / mean) × 100%`

---

## Limitations and Future Work

### Current Limitations

1. **RAPL Scope:**
   - Only measures CPU package energy
   - Ignores external DRAM, GPU, disk I/O, network
   - Intel/AMD CPUs only (not ARM, Apple Silicon)

2. **Constant Power Model:**
   - Rough approximation
   - Does not capture dynamic power states
   - Ignores frequency scaling (DVFS)

3. **Workload Specificity:**
   - Benchmarks focus on gradient compression
   - May not generalize to other data types (text, images, video)

4. **Environmental Factors:**
   - Temperature affects CPU power consumption
   - Thermal throttling can skew measurements
   - No control for background processes

### Future Enhancements

1. **GPU Energy Measurement:**
   - NVIDIA NVML API for GPU power
   - AMD ROCm SMI for AMD GPUs
   - Apple Silicon performance counters

2. **Full-System Energy:**
   - External power meters (USB-C PD monitoring)
   - ACPI battery discharge rate
   - Integration with perf_events

3. **Adaptive Power Models:**
   - Machine learning-based estimation
   - Frequency-aware scaling
   - Workload characterization

4. **Carbon Intensity Integration:**
   - Real-time grid carbon intensity APIs
   - Geographic location-based reporting
   - Time-of-day scheduling optimization

5. **Energy-Driven Compression:**
   - Dynamic quality adjustment based on battery level
   - Power budget-constrained compression
   - Energy-aware rate-distortion optimization

---

## References

1. **Intel RAPL:**
   - Khan, K. N., et al. "RAPL in Action: Experiences in Using RAPL for Power Measurements." *ACM TOMPECS*, 2018.
   - Intel Software Developer's Manual, Vol. 3B, Ch. 14.9: "Platform Power Monitoring"

2. **Energy-Aware Compression:**
   - Deng, Y., et al. "Energy-Efficient Video Compression for Mobile Devices." *IEEE TMM*, 2019.
   - Liu, J., et al. "PowerAdvisor: A Runtime Energy Optimization Framework." *ASPLOS*, 2020.

3. **ML Gradient Compression:**
   - Lin, Y., et al. "Deep Gradient Compression: Reducing Communication Bandwidth for Distributed Training." *ICLR*, 2018.
   - Alistarh, D., et al. "QSGD: Communication-Efficient SGD via Gradient Quantization." *NeurIPS*, 2017.

4. **Carbon Footprint:**
   - Strubell, E., et al. "Energy and Policy Considerations for Deep Learning in NLP." *ACL*, 2019.
   - Henderson, P., et al. "Towards the Systematic Reporting of the Energy Consumption of ML." *Climate Change AI*, 2020.

---

## Appendix: Example Usage

### Basic Energy Measurement

```python
from energy.profiler import EnergyProfiler
from metrics.caq_energy_metric import compute_caqe

# Measure energy for compression
with EnergyProfiler() as profiler:
    compressed_size = compress_file(input_path, output_path)

joules, seconds = profiler.read()
ratio = input_size / compressed_size
caq_e = compute_caqe(ratio, seconds, joules)

print(f"CAQ-E: {caq_e:.4f}")
print(f"Energy: {joules:.2f} J")
print(f"Power: {joules/seconds:.2f} W")
```

### Benchmark with Delta

```python
from experiments.run_energy_benchmark import run_energy_benchmark_suite

datasets = {
    "gradients": Path("energy/datasets/gradients/cifar10_resnet8_gradients.npz")
}

results = run_energy_benchmark_suite(
    datasets=datasets,
    output_path=Path("results/benchmark_results.json"),
    num_runs=3
)

print(f"Mean CAQ-E improvement: {results['summary']['mean_caqe_improvement']:.2f}%")
print(f"Threshold met: {results['summary']['all_thresholds_met']}")
```

### Leaderboard Submission

```json
{
  "submitter": "athanase_lab",
  "dataset": "cifar10_resnet8",
  "codec": "src-adaptive:v0.3.0",
  "compression_ratio": 3.69,
  "cpu_seconds": 0.010,
  "energy_joules": 0.351,
  "caq": 3.68,
  "caq_e": 10.22,
  "device_info": {
    "cpu_model": "Intel Core i5-8265U",
    "cores": 4,
    "threads": 8,
    "base_freq_mhz": 1600
  }
}
```

---

## Runtime Guardrails (Phase H.5.1)

**Added:** 2025-10-17
**Status:** Implemented ✓

### Overview

Phase H.5.1 introduces **runtime guardrails** to protect energy measurements against numeric and statistical anomalies. These guardrails ensure stability, reproducibility, and validity of CAQ-E benchmarks.

### Motivation

Energy measurements can be noisy due to:
- **System interference**: Background processes, thermal throttling
- **Hardware variance**: CPU frequency scaling, power state transitions
- **Statistical outliers**: Anomalous runs skewing averages
- **Numeric instability**: Division by zero, floating-point errors

**Solution**: In-process validation with automatic rejection of unstable runs.

### Guardrail Components

#### 1. Finite Metrics Check

Detects NaN/Inf values in real-time:

```python
from energy.runtime_guard import RuntimeGuard

guard = RuntimeGuard()
is_valid, error = guard.check_finite_metrics({
    "caq_e": 0.238,
    "energy_joules": 10.5,
    "cpu_seconds": 0.5
})
# → (True, None)
```

#### 2. Variance Gate

Rejects runs with excessive cross-run variance using **IQR/median ratio**:

```python
caqe_values = [10.25, 10.24, 10.25, 10.26, 10.25]
passes, stats = guard.check_variance_gate(caqe_values)

# stats = {
#   "variance_percent": 0.097%,  ← Well below 25% threshold
#   "passes": True
# }
```

**Threshold**: IQR/median ≤ 25%

**Why IQR/median?**
- Robust to outliers (unlike standard deviation)
- Percentile-based (not affected by extreme values)
- Validated against Phase H.5 benchmarks (achieved <1% variance)

#### 3. Sanity Range Checks

Validates values are within physically plausible ranges:

| Metric              | Min          | Max        |
|---------------------|--------------|------------|
| Compression Ratio   | 1e-6         | 1e4        |
| CPU Seconds         | 1e-9         | 1e5 (27h)  |
| Energy Joules       | 0.0          | 1e6 (1 MJ) |

#### 4. Rollback Trigger

Detects performance regressions requiring rollback:

```python
guard.create_checkpoint(baseline_caqe, metadata={"version": "v1"})
should_rollback, info = guard.check_rollback_trigger(new_caqe)

if should_rollback:
    print(f"⚠ Performance dropped {info['drop_percent']:.1f}%")
```

**Threshold**: Median CAQ-E drop > 5%

### Integration

Guardrails are automatically applied in:

1. **Benchmark Runner** (`run_energy_benchmark.py`):
   - Per-run validation
   - Aggregate variance check
   - Rollback detection

2. **Leaderboard Update** (`leaderboard_energy_update.py`):
   - Rejects high-variance submissions
   - `--no-variance-gate` option for legacy support

### Example Output

```
======================================================================
BENCHMARK SUMMARY
======================================================================
Datasets Tested: 3
Mean CAQ-E Improvement: 97.41%
Threshold Met: 3/3
Overall Status: ✓ PASS

GUARDRAIL STATUS (Phase H.5.1)
======================================================================
Variance Gate: 3/3 passed
Rollback Checks: 0 triggered
Guardrails Overall: ✓ PASS
```

### Test Coverage

- **Total Tests**: 25 comprehensive tests
- **Test Suite**: `tests/test_runtime_guard.py`
- **Coverage**: 100% of RuntimeGuard public API

```bash
pytest tests/test_runtime_guard.py -v
# → 25 passed in 0.15s
```

### For More Information

See comprehensive documentation: [runtime_guard.md](runtime_guard.md)

---

**End of Energy Model Documentation**

For questions or contributions, see: https://github.com/athanase-matabaro/SRC-Research-Lab

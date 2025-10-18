"""
CAQ-E (Compression-Accuracy-Energy Quotient) Metric

Extends the CAQ metric to incorporate energy consumption, enabling holistic
evaluation of compression algorithms that accounts for both speed and energy.

Formula:
    CAQ-E = compression_ratio / ((cpu_seconds * avg_power_watts) + 1)

Where:
    avg_power_watts = energy_joules / cpu_seconds

Simplified:
    CAQ-E = compression_ratio / (energy_joules + cpu_seconds)

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-16
Phase: H.5 - Energy-Aware Compression
"""

from typing import Dict, Optional
import numpy as np


def compute_caq(
    compression_ratio: float,
    cpu_seconds: float
) -> float:
    """
    Compute CAQ (Compression-Accuracy Quotient) - original Phase H.4 metric.

    Args:
        compression_ratio: Achieved compression ratio.
        cpu_seconds: CPU time in seconds.

    Returns:
        CAQ value (higher is better).

    Formula:
        CAQ = compression_ratio / (cpu_seconds + 1)

    Example:
        >>> compute_caq(2.5, 0.5)
        1.667  # 2.5 / (0.5 + 1)
    """
    if compression_ratio <= 0:
        raise ValueError("Compression ratio must be positive")

    if cpu_seconds < 0:
        raise ValueError("CPU time must be non-negative")

    caq = compression_ratio / (cpu_seconds + 1.0)

    return caq


def compute_caqe(
    compression_ratio: float,
    cpu_seconds: float,
    energy_joules: float
) -> float:
    """
    Compute CAQ-E (Compression-Accuracy-Energy Quotient).

    Args:
        compression_ratio: Achieved compression ratio (output_size / input_size).
        cpu_seconds: CPU time in seconds.
        energy_joules: Energy consumed in joules.

    Returns:
        CAQ-E value (higher is better).

    Formula:
        CAQ-E = compression_ratio / (energy_joules + cpu_seconds)

    Note:
        We add cpu_seconds to the denominator to maintain dimensional balance
        and prevent division by zero for very low-energy operations.

    Example:
        >>> compute_caqe(2.5, 0.5, 17.5)  # 2.5x ratio, 0.5s, 17.5J
        0.1389  # CAQ-E value
    """
    if compression_ratio <= 0:
        raise ValueError("Compression ratio must be positive")

    if cpu_seconds < 0:
        raise ValueError("CPU time must be non-negative")

    if energy_joules < 0:
        raise ValueError("Energy must be non-negative")

    # CAQ-E formula
    caq_e = compression_ratio / (energy_joules + cpu_seconds)

    return caq_e


def compute_caqe_from_power(
    compression_ratio: float,
    cpu_seconds: float,
    avg_power_watts: float
) -> float:
    """
    Compute CAQ-E from average power instead of total energy.

    Args:
        compression_ratio: Achieved compression ratio.
        cpu_seconds: CPU time in seconds.
        avg_power_watts: Average power draw in watts.

    Returns:
        CAQ-E value.

    Note:
        This is equivalent to compute_caqe() with energy = power * time.
    """
    energy_joules = avg_power_watts * cpu_seconds
    return compute_caqe(compression_ratio, cpu_seconds, energy_joules)


def compute_caq_and_caqe(
    compression_ratio: float,
    cpu_seconds: float,
    energy_joules: float
) -> Dict[str, float]:
    """
    Compute both CAQ and CAQ-E metrics.

    Args:
        compression_ratio: Achieved compression ratio.
        cpu_seconds: CPU time in seconds.
        energy_joules: Energy consumed in joules.

    Returns:
        Dictionary with both metrics.

    Example:
        >>> result = compute_caq_and_caqe(2.5, 0.5, 17.5)
        >>> result["caq"]
        1.667  # 2.5 / (0.5 + 1)
        >>> result["caq_e"]
        0.139  # 2.5 / (17.5 + 0.5)
    """
    # CAQ (original metric from Phase H.4)
    caq = compression_ratio / (cpu_seconds + 1.0)

    # CAQ-E (new energy-aware metric)
    caq_e = compute_caqe(compression_ratio, cpu_seconds, energy_joules)

    # Average power
    avg_power = energy_joules / cpu_seconds if cpu_seconds > 0 else 0

    return {
        "caq": caq,
        "caq_e": caq_e,
        "avg_power_watts": avg_power,
        "energy_efficiency": compression_ratio / energy_joules if energy_joules > 0 else 0,
    }


def compute_caqe_delta(
    caqe_adaptive: float,
    caqe_baseline: float
) -> float:
    """
    Compute percentage improvement of adaptive method over baseline.

    Args:
        caqe_adaptive: CAQ-E of adaptive compression.
        caqe_baseline: CAQ-E of baseline compression.

    Returns:
        Percentage improvement (positive = adaptive is better).
        Returns 0.0 if baseline is zero (undefined).

    Example:
        >>> compute_caqe_delta(0.150, 0.125)
        20.0  # 20% improvement
    """
    if caqe_baseline <= 0:
        return 0.0  # Undefined, return 0

    delta = ((caqe_adaptive - caqe_baseline) / caqe_baseline) * 100.0

    return delta


def validate_caqe_threshold(
    caqe_adaptive: float,
    caqe_baseline: float,
    threshold_percent: float = 10.0
) -> bool:
    """
    Check if adaptive method meets CAQ-E improvement threshold.

    Args:
        caqe_adaptive: CAQ-E of adaptive compression.
        caqe_baseline: CAQ-E of baseline compression.
        threshold_percent: Minimum improvement required (default: 10%).

    Returns:
        True if threshold is met, False otherwise.

    Example:
        >>> validate_caqe_threshold(0.150, 0.125, threshold_percent=10.0)
        True  # 20% > 10% threshold
    """
    delta = compute_caqe_delta(caqe_adaptive, caqe_baseline)
    return delta >= threshold_percent


def compute_energy_variance(
    energy_measurements: list,
    relative: bool = True
) -> float:
    """
    Compute variance of energy measurements.

    Args:
        energy_measurements: List of energy values (joules).
        relative: If True, return relative variance (CV%), else absolute stddev.

    Returns:
        Variance (percentage if relative=True, else absolute stddev).
        Returns 0.0 for empty list or single value.

    Example:
        >>> compute_energy_variance([17.5, 17.8, 17.2], relative=True)
        1.73  # 1.73% coefficient of variation
    """
    if len(energy_measurements) == 0:
        return 0.0

    if len(energy_measurements) < 2:
        return 0.0

    arr = np.array(energy_measurements)
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)  # Sample standard deviation

    if relative:
        if mean_val == 0:
            return 0.0
        return (std_val / mean_val) * 100.0  # Coefficient of variation (%)
    else:
        return std_val


def normalize_caqe_by_baseline(
    caqe: float,
    baseline_caqe: float
) -> float:
    """
    Normalize CAQ-E relative to baseline.

    Args:
        caqe: CAQ-E to normalize.
        baseline_caqe: Baseline CAQ-E.

    Returns:
        Normalized CAQ-E (1.0 = baseline, >1.0 = better, <1.0 = worse).

    Example:
        >>> normalize_caqe_by_baseline(0.150, 0.125)
        1.20  # 20% better than baseline
    """
    if baseline_caqe <= 0:
        raise ValueError("Baseline CAQ-E must be positive")

    return caqe / baseline_caqe


def format_caqe_results(
    compression_ratio: float,
    cpu_seconds: float,
    energy_joules: float,
    baseline_caqe: Optional[float] = None
) -> str:
    """
    Format CAQ-E results as human-readable string.

    Args:
        compression_ratio: Achieved compression ratio.
        cpu_seconds: CPU time in seconds.
        energy_joules: Energy consumed in joules.
        baseline_caqe: Optional baseline CAQ-E for comparison.

    Returns:
        Formatted results string.
    """
    results = compute_caq_and_caqe(compression_ratio, cpu_seconds, energy_joules)

    lines = [
        "=" * 60,
        "CAQ-E METRICS",
        "=" * 60,
        f"Compression Ratio: {compression_ratio:.4f}x",
        f"CPU Time: {cpu_seconds:.6f} s",
        f"Energy Consumed: {energy_joules:.4f} J",
        f"Average Power: {results['avg_power_watts']:.2f} W",
        "",
        f"CAQ (original): {results['caq']:.4f}",
        f"CAQ-E (energy-aware): {results['caq_e']:.6f}",
        f"Energy Efficiency: {results['energy_efficiency']:.6f} (ratio/J)",
    ]

    if baseline_caqe is not None:
        delta = compute_caqe_delta(results['caq_e'], baseline_caqe)
        normalized = normalize_caqe_by_baseline(results['caq_e'], baseline_caqe)

        lines.extend([
            "",
            "Comparison to Baseline:",
            f"  Baseline CAQ-E: {baseline_caqe:.6f}",
            f"  Delta: {delta:+.2f}%",
            f"  Normalized: {normalized:.4f}x",
        ])

    lines.append("=" * 60)

    return "\n".join(lines)


# Acceptance criteria constants for Phase H.5
PHASE_H5_CAQE_THRESHOLD = 10.0  # Minimum CAQ-E improvement (%)
PHASE_H5_ENERGY_VARIANCE_MAX = 5.0  # Maximum acceptable variance (%)

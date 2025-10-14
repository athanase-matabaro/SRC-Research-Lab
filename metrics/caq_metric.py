#!/usr/bin/env python3
"""
CAQ Metric (Compression-Accuracy Quotient)

The CAQ metric balances compression ratio with computational cost:
    CAQ = compression_ratio / (cpu_seconds + 1)

Higher CAQ scores indicate better overall performance.
The +1 in denominator prevents division by zero and ensures
even instantaneous compression has a finite score.
"""


def compute_caq(compression_ratio: float, cpu_seconds: float) -> float:
    """
    Compute the Compression-Accuracy Quotient (CAQ).

    Args:
        compression_ratio: Ratio of original size to compressed size (must be > 0)
        cpu_seconds: CPU time in seconds (must be >= 0)

    Returns:
        float: CAQ score

    Raises:
        ValueError: If inputs are invalid
    """
    if compression_ratio <= 0:
        raise ValueError(f"compression_ratio must be > 0, got {compression_ratio}")
    if cpu_seconds < 0:
        raise ValueError(f"cpu_seconds must be >= 0, got {cpu_seconds}")

    return compression_ratio / (cpu_seconds + 1.0)


def compute_variance(values: list) -> float:
    """
    Compute variance percentage across multiple runs.

    Variance = (max - min) / mean * 100

    Args:
        values: List of numeric values

    Returns:
        float: Variance as a percentage

    Raises:
        ValueError: If list is empty or mean is zero
    """
    if not values:
        raise ValueError("Cannot compute variance of empty list")

    mean_val = sum(values) / len(values)
    if mean_val == 0:
        raise ValueError("Cannot compute variance when mean is zero")

    max_val = max(values)
    min_val = min(values)

    return (max_val - min_val) / mean_val * 100.0


if __name__ == "__main__":
    # Example usage
    ratio = 5.63
    cpu_time = 0.26
    caq = compute_caq(ratio, cpu_time)
    print(f"CAQ({ratio}, {cpu_time}) = {caq:.2f}")

    # Example variance
    runs = [5.60, 5.64, 5.65]
    var = compute_variance(runs)
    print(f"Variance({runs}) = {var:.2f}%")

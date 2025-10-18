"""
Energy Utilities - Helper Functions for Energy Profiling

Provides normalization, validation, and analysis utilities for energy measurements.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-16
Phase: H.5 - Energy-Aware Compression
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def normalize_energy(
    joules: float,
    reference_joules: float = 1.0
) -> float:
    """
    Normalize energy consumption relative to a reference value.

    Args:
        joules: Measured energy in joules.
        reference_joules: Reference energy for normalization (default: 1.0).

    Returns:
        Normalized energy (joules / reference_joules).
    """
    if reference_joules <= 0:
        raise ValueError("Reference joules must be positive")

    return joules / reference_joules


def validate_energy_reading(
    joules: float,
    seconds: float,
    max_power_watts: float = 500.0,
    min_power_watts: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate energy reading for plausibility.

    Args:
        joules: Measured energy.
        seconds: Elapsed time.
        max_power_watts: Maximum plausible power draw.
        min_power_watts: Minimum plausible power draw.

    Returns:
        Tuple of (is_valid, error_message).
        error_message is None if valid.
    """
    if joules < 0:
        return (False, f"Negative energy reading: {joules} J")

    if seconds <= 0:
        return (False, f"Invalid elapsed time: {seconds} s")

    avg_power = joules / seconds

    if avg_power > max_power_watts:
        return (
            False,
            f"Implausibly high power: {avg_power:.1f}W (max: {max_power_watts}W)"
        )

    if avg_power < min_power_watts:
        return (
            False,
            f"Implausibly low power: {avg_power:.1f}W (min: {min_power_watts}W)"
        )

    return (True, None)


def compute_energy_efficiency(
    compression_ratio: float,
    joules: float
) -> float:
    """
    Compute energy efficiency: compression ratio per joule.

    Args:
        compression_ratio: Achieved compression ratio.
        joules: Energy consumed.

    Returns:
        Efficiency (ratio / joule).
    """
    if joules <= 0:
        raise ValueError("Energy must be positive")

    return compression_ratio / joules


def compute_variance(values: List[float]) -> float:
    """
    Compute variance of energy measurements.

    Args:
        values: List of energy measurements.

    Returns:
        Variance as percentage of mean.
    """
    if len(values) < 2:
        return 0.0

    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)

    if mean == 0:
        return 0.0

    return (std / mean) * 100.0  # Percentage


def save_energy_profile(
    output_path: Path,
    joules: float,
    seconds: float,
    method: str,
    cpu_info: Dict[str, str],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save energy profile to JSON file.

    Args:
        output_path: Output file path.
        joules: Energy consumed.
        seconds: Elapsed time.
        method: Measurement method (rapl/constant).
        cpu_info: CPU information dictionary.
        metadata: Additional metadata to include.
    """
    avg_power = joules / seconds if seconds > 0 else 0

    profile = {
        "energy_joules": joules,
        "elapsed_seconds": seconds,
        "avg_power_watts": avg_power,
        "measurement_method": method,
        "cpu_info": cpu_info,
    }

    if metadata:
        profile["metadata"] = metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)


def load_energy_profile(input_path: Path) -> Dict:
    """
    Load energy profile from JSON file.

    Args:
        input_path: Input file path.

    Returns:
        Dictionary with energy profile data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Energy profile not found: {input_path}")

    with open(input_path, 'r') as f:
        profile = json.load(f)

    # Validate required fields
    required_fields = ["energy_joules", "elapsed_seconds", "measurement_method"]
    for field in required_fields:
        if field not in profile:
            raise ValueError(f"Missing required field: {field}")

    return profile


def compare_energy_profiles(
    profile1: Dict,
    profile2: Dict
) -> Dict[str, float]:
    """
    Compare two energy profiles.

    Args:
        profile1: First energy profile.
        profile2: Second energy profile (baseline).

    Returns:
        Dictionary with comparison metrics.
    """
    joules1 = profile1["energy_joules"]
    joules2 = profile2["energy_joules"]

    seconds1 = profile1["elapsed_seconds"]
    seconds2 = profile2["elapsed_seconds"]

    power1 = joules1 / seconds1 if seconds1 > 0 else 0
    power2 = joules2 / seconds2 if seconds2 > 0 else 0

    energy_delta = ((joules1 - joules2) / joules2) * 100 if joules2 > 0 else 0
    power_delta = ((power1 - power2) / power2) * 100 if power2 > 0 else 0
    time_delta = ((seconds1 - seconds2) / seconds2) * 100 if seconds2 > 0 else 0

    return {
        "energy_delta_percent": energy_delta,
        "power_delta_percent": power_delta,
        "time_delta_percent": time_delta,
        "energy_ratio": joules1 / joules2 if joules2 > 0 else 0,
    }


def estimate_carbon_footprint(
    joules: float,
    carbon_intensity_g_per_kwh: float = 500.0
) -> float:
    """
    Estimate CO2 emissions from energy consumption.

    Args:
        joules: Energy consumed.
        carbon_intensity_g_per_kwh: Carbon intensity of electricity grid
                                    (grams CO2 per kWh). Default: 500g/kWh
                                    (typical mixed grid).

    Returns:
        Estimated CO2 emissions in grams.

    Note:
        Carbon intensity varies by region and time:
        - Coal-heavy: 800-1000 g/kWh
        - Mixed grid: 400-600 g/kWh
        - Renewables: 50-200 g/kWh
    """
    # Convert joules to kWh
    kwh = joules / 3_600_000.0  # 1 kWh = 3,600,000 joules

    # Calculate CO2 emissions
    co2_grams = kwh * carbon_intensity_g_per_kwh

    return co2_grams


def format_energy_report(
    joules: float,
    seconds: float,
    method: str,
    cpu_info: Dict[str, str],
    include_carbon: bool = True
) -> str:
    """
    Format energy measurement as human-readable report.

    Args:
        joules: Energy consumed.
        seconds: Elapsed time.
        method: Measurement method.
        cpu_info: CPU information.
        include_carbon: Whether to include carbon footprint estimate.

    Returns:
        Formatted report string.
    """
    avg_power = joules / seconds if seconds > 0 else 0

    report = [
        "=" * 60,
        "ENERGY PROFILE REPORT",
        "=" * 60,
        f"Energy Consumed: {joules:.4f} J",
        f"Elapsed Time: {seconds:.6f} s",
        f"Average Power: {avg_power:.2f} W",
        f"Measurement Method: {method}",
        "",
        "CPU Information:",
        f"  Model: {cpu_info.get('model', 'unknown')}",
        f"  Cores: {cpu_info.get('cores', 'unknown')}",
        f"  Threads: {cpu_info.get('threads', 'unknown')}",
        f"  Base Frequency: {cpu_info.get('base_freq_mhz', 'unknown')} MHz",
    ]

    if include_carbon:
        co2_grams = estimate_carbon_footprint(joules)
        report.extend([
            "",
            "Environmental Impact:",
            f"  Estimated CO2: {co2_grams:.6f} g (at 500g/kWh grid intensity)",
        ])

    report.append("=" * 60)

    return "\n".join(report)

"""
Energy Profiler Package - CPU Energy Measurement

Provides offline CPU energy measurement using Intel RAPL or fallback models.

Author: Athanase Nshombo (Matabaro)
Phase: H.5 - Energy-Aware Compression
"""

from .energy_profiler import EnergyProfiler, measure_energy
from .energy_utils import (
    normalize_energy,
    validate_energy_reading,
    compute_energy_efficiency,
    compute_variance,
    save_energy_profile,
    load_energy_profile,
    compare_energy_profiles,
    estimate_carbon_footprint,
    format_energy_report,
)

__all__ = [
    "EnergyProfiler",
    "measure_energy",
    "normalize_energy",
    "validate_energy_reading",
    "compute_energy_efficiency",
    "compute_variance",
    "save_energy_profile",
    "load_energy_profile",
    "compare_energy_profiles",
    "estimate_carbon_footprint",
    "format_energy_report",
]

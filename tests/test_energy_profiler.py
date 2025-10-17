"""
Unit Tests for Energy Profiler

Tests energy measurement infrastructure including RAPL detection,
fallback model, validation, and context manager functionality.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Phase: H.5 - Energy-Aware Compression
"""

import sys
import time
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from energy.profiler import (
    EnergyProfiler,
    measure_energy,
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


class TestEnergyProfilerConstantMode:
    """Test energy profiler in constant power mode (fallback)."""

    def test_constant_mode_basic(self):
        """Test basic energy measurement with constant power."""
        profiler = EnergyProfiler(quiet=True)

        # Should default to constant mode if RAPL not available
        assert profiler.method in ["constant", "rapl"]

        if profiler.method == "constant":
            assert profiler.constant_power == EnergyProfiler.DEFAULT_POWER_WATTS

    def test_context_manager(self):
        """Test context manager functionality."""
        with EnergyProfiler(quiet=True) as profiler:
            time.sleep(0.01)  # 10ms work

        joules, seconds = profiler.read()

        assert joules > 0, "Energy should be positive"
        assert seconds >= 0.01, "Time should be at least 10ms"
        assert seconds < 1.0, "Time should be reasonable"

    def test_multiple_reads(self):
        """Test multiple energy readings."""
        profiler = EnergyProfiler(quiet=True)
        profiler.start()
        time.sleep(0.01)
        profiler.stop()

        joules1, seconds1 = profiler.read()
        joules2, seconds2 = profiler.read()

        # Readings should be consistent
        assert joules1 == joules2
        assert seconds1 == seconds2

    def test_custom_power(self):
        """Test custom power consumption value."""
        custom_power = 50.0
        profiler = EnergyProfiler(constant_power=custom_power, quiet=True)

        profiler.start()
        time.sleep(0.1)  # 100ms
        profiler.stop()

        joules, seconds = profiler.read()

        # Energy should be roughly custom_power * seconds
        expected_joules = custom_power * seconds
        assert abs(joules - expected_joules) < 1.0  # Within 1 joule


class TestEnergyProfilerRAPLMode:
    """Test energy profiler with RAPL (if available)."""

    def test_rapl_detection(self):
        """Test RAPL detection and initialization."""
        profiler = EnergyProfiler(quiet=True)

        # RAPL is detected if available, otherwise falls back to constant
        if (Path("/sys/class/powercap/intel-rapl") / "intel-rapl:0").exists():
            # If RAPL directory exists, we expect it to be detected
            # (unless permission issues prevent reading)
            assert profiler.method in ["rapl", "constant"]
        else:
            # If RAPL doesn't exist, must fall back to constant
            assert profiler.method == "constant"

    @pytest.mark.skipif(
        not (Path("/sys/class/powercap/intel-rapl") / "intel-rapl:0").exists(),
        reason="RAPL not available on this system"
    )
    def test_rapl_measurement(self):
        """Test actual RAPL energy measurement."""
        with EnergyProfiler(quiet=True) as profiler:
            # Do some CPU work
            _ = sum(i**2 for i in range(100000))

        joules, seconds = profiler.read()

        assert joules > 0
        assert seconds > 0

        # Power should be reasonable (1-500W)
        avg_power = joules / seconds
        assert 1.0 <= avg_power <= 500.0


class TestMeasureEnergyFunction:
    """Test measure_energy convenience function."""

    def test_measure_energy_basic(self):
        """Test energy measurement function."""

        def sample_function():
            time.sleep(0.01)
            return 42

        result, joules, seconds = measure_energy(sample_function, quiet=True)

        assert result == 42
        assert joules > 0
        assert seconds >= 0.01

    def test_measure_energy_with_args(self):
        """Test function with arguments."""

        def add(a, b):
            return a + b

        result, joules, seconds = measure_energy(add, 10, 20, quiet=True)

        assert result == 30
        assert joules > 0
        assert seconds > 0


class TestEnergyUtils:
    """Test energy utility functions."""

    def test_normalize_energy(self):
        """Test energy normalization."""
        joules = 10.0
        reference = 5.0

        normalized = normalize_energy(joules, reference)
        assert normalized == 2.0

        # Test with default reference
        normalized_default = normalize_energy(joules)
        assert normalized_default == 10.0

    def test_normalize_energy_invalid(self):
        """Test normalization with invalid reference."""
        with pytest.raises(ValueError):
            normalize_energy(10.0, 0.0)

        with pytest.raises(ValueError):
            normalize_energy(10.0, -5.0)

    def test_validate_energy_reading_valid(self):
        """Test validation of valid energy readings."""
        joules = 10.0
        seconds = 1.0

        is_valid, error_msg = validate_energy_reading(joules, seconds)
        assert is_valid
        assert error_msg is None

    def test_validate_energy_reading_negative(self):
        """Test validation with negative energy."""
        is_valid, error_msg = validate_energy_reading(-5.0, 1.0)
        assert not is_valid
        assert "Negative energy" in error_msg

    def test_validate_energy_reading_high_power(self):
        """Test validation with implausibly high power."""
        # 1000J in 1s = 1000W (exceeds default 500W max)
        is_valid, error_msg = validate_energy_reading(1000.0, 1.0)
        assert not is_valid
        assert "high power" in error_msg

    def test_validate_energy_reading_low_power(self):
        """Test validation with implausibly low power."""
        # 0.1J in 1s = 0.1W (below default 1W min)
        is_valid, error_msg = validate_energy_reading(0.1, 1.0)
        assert not is_valid
        assert "low power" in error_msg

    def test_compute_energy_efficiency(self):
        """Test energy efficiency calculation."""
        compression_ratio = 2.0
        joules = 10.0

        efficiency = compute_energy_efficiency(compression_ratio, joules)
        assert efficiency == 0.2

    def test_compute_energy_efficiency_invalid(self):
        """Test efficiency with invalid energy."""
        with pytest.raises(ValueError):
            compute_energy_efficiency(2.0, 0.0)

        with pytest.raises(ValueError):
            compute_energy_efficiency(2.0, -5.0)

    def test_compute_variance(self):
        """Test variance computation."""
        values = [10.0, 12.0, 11.0, 13.0, 9.0]
        variance = compute_variance(values)

        # Should be around 15% variance
        assert 10.0 <= variance <= 20.0

    def test_compute_variance_single_value(self):
        """Test variance with single value."""
        variance = compute_variance([10.0])
        assert variance == 0.0

    def test_compute_variance_zero_mean(self):
        """Test variance with zero mean."""
        variance = compute_variance([0.0, 0.0, 0.0])
        assert variance == 0.0


class TestEnergyProfileIO:
    """Test energy profile save/load functionality."""

    def test_save_and_load_profile(self):
        """Test saving and loading energy profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profile.json"

            cpu_info = {
                "model": "Test CPU",
                "cores": "4",
                "threads": "8",
                "base_freq_mhz": "2000"
            }

            save_energy_profile(
                output_path,
                joules=10.5,
                seconds=2.0,
                method="constant",
                cpu_info=cpu_info,
                metadata={"test": "data"}
            )

            assert output_path.exists()

            profile = load_energy_profile(output_path)

            assert profile["energy_joules"] == 10.5
            assert profile["elapsed_seconds"] == 2.0
            assert profile["measurement_method"] == "constant"
            assert profile["cpu_info"]["model"] == "Test CPU"
            assert profile["metadata"]["test"] == "data"

    def test_load_missing_profile(self):
        """Test loading non-existent profile."""
        with pytest.raises(FileNotFoundError):
            load_energy_profile(Path("/nonexistent/profile.json"))

    def test_load_invalid_profile(self):
        """Test loading profile with missing required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "invalid.json"
            output_path.write_text('{"invalid": "data"}')

            with pytest.raises(ValueError, match="Missing required field"):
                load_energy_profile(output_path)


class TestCompareProfiles:
    """Test energy profile comparison."""

    def test_compare_profiles(self):
        """Test comparison of two energy profiles."""
        profile1 = {
            "energy_joules": 15.0,
            "elapsed_seconds": 2.0
        }
        profile2 = {
            "energy_joules": 10.0,
            "elapsed_seconds": 2.0
        }

        comparison = compare_energy_profiles(profile1, profile2)

        assert comparison["energy_delta_percent"] == 50.0  # 50% increase
        assert comparison["energy_ratio"] == 1.5
        assert comparison["time_delta_percent"] == 0.0

    def test_compare_profiles_different_times(self):
        """Test comparison with different execution times."""
        profile1 = {
            "energy_joules": 20.0,
            "elapsed_seconds": 4.0
        }
        profile2 = {
            "energy_joules": 10.0,
            "elapsed_seconds": 2.0
        }

        comparison = compare_energy_profiles(profile1, profile2)

        # Energy doubled, time doubled, so power is the same
        assert comparison["energy_delta_percent"] == 100.0
        assert comparison["time_delta_percent"] == 100.0
        assert comparison["power_delta_percent"] == 0.0


class TestCarbonFootprint:
    """Test carbon footprint estimation."""

    def test_estimate_carbon_basic(self):
        """Test basic carbon footprint calculation."""
        # 3,600,000 joules = 1 kWh
        # 1 kWh * 500 g/kWh = 500g CO2
        joules = 3_600_000.0
        co2_grams = estimate_carbon_footprint(joules)

        assert abs(co2_grams - 500.0) < 1.0

    def test_estimate_carbon_custom_intensity(self):
        """Test carbon footprint with custom grid intensity."""
        joules = 3_600_000.0
        co2_grams = estimate_carbon_footprint(joules, carbon_intensity_g_per_kwh=1000.0)

        assert abs(co2_grams - 1000.0) < 1.0

    def test_estimate_carbon_small_energy(self):
        """Test carbon footprint with small energy values."""
        joules = 100.0  # 100J = 0.0000278 kWh
        co2_grams = estimate_carbon_footprint(joules)

        # Should be very small
        assert 0.0 < co2_grams < 0.1


class TestEnergyReport:
    """Test energy report formatting."""

    def test_format_energy_report_basic(self):
        """Test basic energy report formatting."""
        cpu_info = {
            "model": "Test CPU",
            "cores": "4",
            "threads": "8",
            "base_freq_mhz": "2000"
        }

        report = format_energy_report(
            joules=10.0,
            seconds=2.0,
            method="constant",
            cpu_info=cpu_info,
            include_carbon=False
        )

        assert "ENERGY PROFILE REPORT" in report
        assert "10.0000 J" in report
        assert "2.000000 s" in report
        assert "5.00 W" in report
        assert "Test CPU" in report

    def test_format_energy_report_with_carbon(self):
        """Test energy report with carbon footprint."""
        cpu_info = {
            "model": "Test CPU",
            "cores": "4",
            "threads": "8",
            "base_freq_mhz": "2000"
        }

        report = format_energy_report(
            joules=10.0,
            seconds=2.0,
            method="constant",
            cpu_info=cpu_info,
            include_carbon=True
        )

        assert "CO2" in report
        assert "Environmental Impact" in report


class TestCPUInfo:
    """Test CPU information retrieval."""

    def test_get_cpu_info(self):
        """Test CPU info retrieval."""
        cpu_info = EnergyProfiler.get_cpu_info()

        assert "model" in cpu_info
        assert "cores" in cpu_info
        assert "threads" in cpu_info
        assert "base_freq_mhz" in cpu_info

        # Sanity checks
        cores = int(cpu_info["cores"])
        threads = int(cpu_info["threads"])
        assert cores >= 1
        assert threads >= cores  # Threads should be >= cores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

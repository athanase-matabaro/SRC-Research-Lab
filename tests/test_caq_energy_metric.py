"""
Unit Tests for CAQ-E (Energy-Aware) Metric

Tests CAQ-E computation, validation, and delta calculations.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Phase: H.5 - Energy-Aware Compression
"""

import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.caq_energy_metric import (
    compute_caq,
    compute_caqe,
    compute_caq_and_caqe,
    compute_caqe_delta,
    validate_caqe_threshold,
    compute_energy_variance,
    PHASE_H5_CAQE_THRESHOLD,
)


class TestCAQMetric:
    """Test CAQ (base) metric computation."""

    def test_compute_caq_basic(self):
        """Test basic CAQ computation."""
        compression_ratio = 2.0
        cpu_seconds = 1.0

        caq = compute_caq(compression_ratio, cpu_seconds)
        assert caq == 1.0  # 2.0 / (1.0 + 1.0)

    def test_compute_caq_zero_time(self):
        """Test CAQ with zero CPU time."""
        compression_ratio = 2.0
        cpu_seconds = 0.0

        caq = compute_caq(compression_ratio, cpu_seconds)
        assert caq == 2.0  # 2.0 / (0.0 + 1.0)

    def test_compute_caq_invalid_ratio(self):
        """Test CAQ with invalid compression ratio."""
        with pytest.raises(ValueError, match="Compression ratio must be positive"):
            compute_caq(0.0, 1.0)

        with pytest.raises(ValueError, match="Compression ratio must be positive"):
            compute_caq(-1.0, 1.0)

    def test_compute_caq_invalid_time(self):
        """Test CAQ with invalid CPU time."""
        with pytest.raises(ValueError, match="CPU time must be non-negative"):
            compute_caq(2.0, -1.0)


class TestCAQEMetric:
    """Test CAQ-E (energy-aware) metric computation."""

    def test_compute_caqe_basic(self):
        """Test basic CAQ-E computation."""
        compression_ratio = 2.0
        cpu_seconds = 1.0
        energy_joules = 10.0

        caq_e = compute_caqe(compression_ratio, cpu_seconds, energy_joules)
        expected = 2.0 / (10.0 + 1.0)  # 2.0 / 11.0 ≈ 0.1818
        assert abs(caq_e - expected) < 1e-6

    def test_compute_caqe_zero_energy(self):
        """Test CAQ-E with zero energy (edge case)."""
        compression_ratio = 2.0
        cpu_seconds = 1.0
        energy_joules = 0.0

        caq_e = compute_caqe(compression_ratio, cpu_seconds, energy_joules)
        expected = 2.0 / 1.0  # 2.0 / (0.0 + 1.0)
        assert caq_e == expected

    def test_compute_caqe_high_energy(self):
        """Test CAQ-E with high energy consumption."""
        compression_ratio = 2.0
        cpu_seconds = 0.1
        energy_joules = 100.0

        caq_e = compute_caqe(compression_ratio, cpu_seconds, energy_joules)
        expected = 2.0 / (100.0 + 0.1)  # ≈ 0.0199
        assert abs(caq_e - expected) < 1e-6

    def test_compute_caqe_invalid_ratio(self):
        """Test CAQ-E with invalid compression ratio."""
        with pytest.raises(ValueError, match="Compression ratio must be positive"):
            compute_caqe(0.0, 1.0, 10.0)

        with pytest.raises(ValueError, match="Compression ratio must be positive"):
            compute_caqe(-1.0, 1.0, 10.0)

    def test_compute_caqe_invalid_time(self):
        """Test CAQ-E with invalid CPU time."""
        with pytest.raises(ValueError, match="CPU time must be non-negative"):
            compute_caqe(2.0, -1.0, 10.0)

    def test_compute_caqe_invalid_energy(self):
        """Test CAQ-E with invalid energy."""
        with pytest.raises(ValueError, match="Energy must be non-negative"):
            compute_caqe(2.0, 1.0, -10.0)


class TestCAQAndCAQE:
    """Test combined CAQ and CAQ-E computation."""

    def test_compute_caq_and_caqe(self):
        """Test computing both CAQ and CAQ-E."""
        compression_ratio = 3.0
        cpu_seconds = 0.5
        energy_joules = 5.0

        metrics = compute_caq_and_caqe(compression_ratio, cpu_seconds, energy_joules)

        assert "caq" in metrics
        assert "caq_e" in metrics
        assert "avg_power_watts" in metrics
        assert "energy_efficiency" in metrics

        # Verify calculations
        expected_caq = 3.0 / (0.5 + 1.0)  # 2.0
        expected_caq_e = 3.0 / (5.0 + 0.5)  # ≈ 0.545
        expected_power = 5.0 / 0.5  # 10.0 W
        expected_efficiency = 3.0 / 5.0  # 0.6

        assert abs(metrics["caq"] - expected_caq) < 1e-6
        assert abs(metrics["caq_e"] - expected_caq_e) < 1e-6
        assert abs(metrics["avg_power_watts"] - expected_power) < 1e-6
        assert abs(metrics["energy_efficiency"] - expected_efficiency) < 1e-6

    def test_compute_caq_and_caqe_zero_time(self):
        """Test with zero CPU time (edge case)."""
        metrics = compute_caq_and_caqe(2.0, 0.0, 10.0)

        assert metrics["caq"] == 2.0
        assert metrics["caq_e"] == 2.0 / 10.0
        assert metrics["avg_power_watts"] == 0.0  # Undefined, returns 0


class TestCAQEDelta:
    """Test CAQ-E delta (improvement) calculation."""

    def test_compute_caqe_delta_improvement(self):
        """Test delta with improvement."""
        adaptive_caqe = 1.5
        baseline_caqe = 1.0

        delta = compute_caqe_delta(adaptive_caqe, baseline_caqe)
        assert delta == 50.0  # 50% improvement

    def test_compute_caqe_delta_regression(self):
        """Test delta with regression."""
        adaptive_caqe = 0.8
        baseline_caqe = 1.0

        delta = compute_caqe_delta(adaptive_caqe, baseline_caqe)
        assert abs(delta - (-20.0)) < 1e-6  # 20% regression (with tolerance)

    def test_compute_caqe_delta_no_change(self):
        """Test delta with no change."""
        adaptive_caqe = 1.0
        baseline_caqe = 1.0

        delta = compute_caqe_delta(adaptive_caqe, baseline_caqe)
        assert delta == 0.0

    def test_compute_caqe_delta_zero_baseline(self):
        """Test delta with zero baseline."""
        delta = compute_caqe_delta(1.5, 0.0)
        assert delta == 0.0  # Undefined, returns 0

    def test_compute_caqe_delta_large_improvement(self):
        """Test delta with large improvement."""
        adaptive_caqe = 10.0
        baseline_caqe = 1.0

        delta = compute_caqe_delta(adaptive_caqe, baseline_caqe)
        assert delta == 900.0  # 900% improvement


class TestCAQEThresholdValidation:
    """Test CAQ-E threshold validation."""

    def test_validate_threshold_met(self):
        """Test threshold validation when met."""
        adaptive_caqe = 1.5
        baseline_caqe = 1.0
        threshold = 10.0  # 10% minimum

        is_met = validate_caqe_threshold(adaptive_caqe, baseline_caqe, threshold)
        assert is_met  # 50% > 10%

    def test_validate_threshold_not_met(self):
        """Test threshold validation when not met."""
        adaptive_caqe = 1.05
        baseline_caqe = 1.0
        threshold = 10.0  # 10% minimum

        is_met = validate_caqe_threshold(adaptive_caqe, baseline_caqe, threshold)
        assert not is_met  # 5% < 10%

    def test_validate_threshold_exactly_met(self):
        """Test threshold validation at exact boundary."""
        adaptive_caqe = 1.1
        baseline_caqe = 1.0
        threshold = 10.0  # 10% minimum

        is_met = validate_caqe_threshold(adaptive_caqe, baseline_caqe, threshold)
        assert is_met  # Exactly 10%

    def test_validate_threshold_regression(self):
        """Test threshold validation with regression."""
        adaptive_caqe = 0.9
        baseline_caqe = 1.0
        threshold = 10.0

        is_met = validate_caqe_threshold(adaptive_caqe, baseline_caqe, threshold)
        assert not is_met  # -10% regression

    def test_validate_default_threshold(self):
        """Test validation with default Phase H.5 threshold."""
        adaptive_caqe = 1.2
        baseline_caqe = 1.0

        is_met = validate_caqe_threshold(adaptive_caqe, baseline_caqe)
        assert is_met  # 20% > 10% (default)

        # Verify default threshold
        assert PHASE_H5_CAQE_THRESHOLD == 10.0


class TestEnergyVariance:
    """Test energy variance computation."""

    def test_compute_energy_variance_absolute(self):
        """Test absolute energy variance."""
        energy_values = [10.0, 12.0, 11.0, 13.0, 9.0]

        variance = compute_energy_variance(energy_values, relative=False)

        # Mean = 11.0, variance should be around 1.4-1.6
        assert 1.0 <= variance <= 2.0

    def test_compute_energy_variance_relative(self):
        """Test relative energy variance (percentage)."""
        energy_values = [10.0, 12.0, 11.0, 13.0, 9.0]

        variance_pct = compute_energy_variance(energy_values, relative=True)

        # Should be around 12-15% of mean
        assert 10.0 <= variance_pct <= 20.0

    def test_compute_energy_variance_zero(self):
        """Test variance with identical values."""
        energy_values = [10.0, 10.0, 10.0]

        variance = compute_energy_variance(energy_values)
        assert variance == 0.0

    def test_compute_energy_variance_single_value(self):
        """Test variance with single value."""
        variance = compute_energy_variance([10.0])
        assert variance == 0.0

    def test_compute_energy_variance_empty(self):
        """Test variance with empty list."""
        variance = compute_energy_variance([])
        assert variance == 0.0


class TestPhaseH5Requirements:
    """Test Phase H.5 specific requirements."""

    def test_caqe_better_than_caq_for_efficient_compression(self):
        """
        Test that CAQ-E rewards energy-efficient compression.

        Scenario: Two methods with same compression ratio and time,
        but different energy consumption.
        """
        compression_ratio = 2.0
        cpu_seconds = 1.0

        # Method A: High energy
        energy_a = 100.0
        caq_a = compute_caq(compression_ratio, cpu_seconds)
        caqe_a = compute_caqe(compression_ratio, cpu_seconds, energy_a)

        # Method B: Low energy (more efficient)
        energy_b = 10.0
        caq_b = compute_caq(compression_ratio, cpu_seconds)
        caqe_b = compute_caqe(compression_ratio, cpu_seconds, energy_b)

        # CAQ should be same (energy-agnostic)
        assert caq_a == caq_b

        # CAQ-E should favor low-energy method
        assert caqe_b > caqe_a

    def test_phase_h5_threshold_10_percent(self):
        """Test that Phase H.5 threshold is 10%."""
        assert PHASE_H5_CAQE_THRESHOLD == 10.0

    def test_realistic_gradient_compression_scenario(self):
        """
        Test realistic gradient compression scenario.

        Based on Phase H.5 benchmark results:
        - Baseline: ratio=1.32, time=0.006s, energy=0.206J
        - Adaptive: ratio=3.69, time=0.010s, energy=0.351J
        """
        # Baseline
        baseline_metrics = compute_caq_and_caqe(1.32, 0.006, 0.206)

        # Adaptive
        adaptive_metrics = compute_caq_and_caqe(3.69, 0.010, 0.351)

        # Compute delta
        delta_caqe = compute_caqe_delta(
            adaptive_metrics["caq_e"],
            baseline_metrics["caq_e"]
        )

        # Should show significant improvement (>10%)
        assert delta_caqe > PHASE_H5_CAQE_THRESHOLD

        # Adaptive should have higher CAQ-E despite higher energy use
        # (because compression ratio improvement dominates)
        assert adaptive_metrics["caq_e"] > baseline_metrics["caq_e"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_values(self):
        """Test with very small energy/time values."""
        metrics = compute_caq_and_caqe(
            compression_ratio=1.5,
            cpu_seconds=0.0001,
            energy_joules=0.001
        )

        assert metrics["caq"] > 0
        assert metrics["caq_e"] > 0
        assert metrics["avg_power_watts"] > 0

    def test_very_large_values(self):
        """Test with very large energy/time values."""
        metrics = compute_caq_and_caqe(
            compression_ratio=10.0,
            cpu_seconds=1000.0,
            energy_joules=100000.0
        )

        assert metrics["caq"] > 0
        assert metrics["caq_e"] > 0
        assert metrics["avg_power_watts"] > 0

    def test_high_compression_low_energy(self):
        """Test ideal scenario: high compression, low energy."""
        metrics = compute_caq_and_caqe(
            compression_ratio=10.0,
            cpu_seconds=0.1,
            energy_joules=1.0
        )

        # Should have excellent CAQ-E
        assert metrics["caq_e"] > 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Robustness Tests for Phase H.5

Tests edge cases and guardrails for energy-aware compression.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Phase: H.5 Robustness Hardening
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.caq_energy_metric import compute_caqe, compute_caq_and_caqe


class TestEpsilonProtection:
    """Test epsilon protection in CAQ-E computation."""

    def test_caqe_zero_time_zero_energy(self):
        """Test CAQ-E with zero time and zero energy (edge case)."""
        # Should not raise division by zero due to epsilon protection
        caq_e = compute_caqe(compression_ratio=2.0, cpu_seconds=0.0, energy_joules=0.0)

        # Should return finite value
        assert np.isfinite(caq_e)
        assert caq_e > 0

    def test_caqe_extremely_small_values(self):
        """Test CAQ-E with extremely small denominator."""
        caq_e = compute_caqe(
            compression_ratio=2.0,
            cpu_seconds=1e-12,
            energy_joules=1e-12
        )

        # Should not overflow or return inf
        assert np.isfinite(caq_e)
        assert caq_e > 0

    def test_caqe_large_compression_ratio(self):
        """Test CAQ-E with very large compression ratio."""
        caq_e = compute_caqe(
            compression_ratio=1000.0,
            cpu_seconds=0.001,
            energy_joules=0.001
        )

        assert np.isfinite(caq_e)
        assert caq_e > 0

    def test_caqe_result_always_finite(self):
        """Test that CAQ-E always returns finite values."""
        test_cases = [
            (1.5, 0.0, 0.0),  # Zero denominator
            (2.0, 1e-15, 1e-15),  # Extremely small
            (10.0, 1.0, 100.0),  # Normal case
            (0.5, 10.0, 1000.0),  # Poor compression
        ]

        for ratio, time, energy in test_cases:
            caq_e = compute_caqe(ratio, time, energy)
            assert np.isfinite(caq_e), f"Non-finite CAQ-E for ({ratio}, {time}, {energy})"


class TestZeroScaleHandling:
    """Test handling of zero-scale tensors in quantization."""

    def test_all_zeros_tensor(self):
        """Test compression of all-zeros tensor."""
        # Simulate all-zeros gradient
        gradient = np.zeros((100, 100), dtype=np.float32)

        # Import compression functions
        from experiments.run_energy_benchmark import compress_gradient_baseline

        # Should not crash
        compressed = compress_gradient_baseline(gradient)

        assert len(compressed) > 0
        assert isinstance(compressed, bytes)

    def test_single_outlier_tensor(self):
        """Test tensor with all zeros except one outlier."""
        gradient = np.zeros((100, 100), dtype=np.float32)
        gradient[50, 50] = 1000.0  # Single large outlier

        from experiments.run_energy_benchmark import compress_gradient_baseline

        compressed = compress_gradient_baseline(gradient)

        assert len(compressed) > 0
        # Compression ratio should be finite
        ratio = gradient.nbytes / len(compressed)
        assert np.isfinite(ratio)

    def test_extreme_range_tensor(self):
        """Test tensor with extreme value range."""
        gradient = np.random.randn(100, 100).astype(np.float32) * 1e-10
        gradient[0, 0] = 1e10  # Extreme outlier

        from experiments.run_energy_benchmark import compress_gradient_baseline

        compressed = compress_gradient_baseline(gradient)

        assert len(compressed) > 0
        ratio = gradient.nbytes / len(compressed)
        assert np.isfinite(ratio)


class TestSanityChecks:
    """Test runtime sanity checks and guardrails."""

    def test_compression_ratio_sanity(self):
        """Test that compression ratio is always > 0."""
        from experiments.run_energy_benchmark import compress_gradient_baseline

        gradient = np.random.randn(100, 100).astype(np.float32)
        compressed = compress_gradient_baseline(gradient)

        ratio = gradient.nbytes / len(compressed)

        assert ratio > 0
        assert np.isfinite(ratio)
        # Ratio should be reasonable (between 0.1 and 100)
        assert 0.1 < ratio < 100.0

    def test_energy_measurement_sanity(self):
        """Test that energy measurements are reasonable."""
        from energy.profiler import EnergyProfiler
        import time

        with EnergyProfiler(quiet=True) as profiler:
            time.sleep(0.01)  # 10ms work

        joules, seconds = profiler.read()

        # Sanity checks
        assert joules >= 0, "Energy must be non-negative"
        assert seconds >= 0.01, "Time should be at least 10ms"
        assert np.isfinite(joules), "Energy must be finite"
        assert np.isfinite(seconds), "Time must be finite"

        # Power should be reasonable (1W - 500W)
        if seconds > 0:
            power = joules / seconds
            assert 0.1 < power < 1000.0, f"Unreasonable power: {power}W"

    def test_caqe_range_sanity(self):
        """Test that CAQ-E values are in reasonable range."""
        # Typical values
        test_cases = [
            (2.0, 1.0, 10.0),  # Normal case
            (1.5, 0.5, 5.0),   # Low energy
            (5.0, 2.0, 50.0),  # High compression
        ]

        for ratio, time, energy in test_cases:
            caq_e = compute_caqe(ratio, time, energy)

            # CAQ-E should be positive and finite
            assert caq_e > 0
            assert np.isfinite(caq_e)
            # Typical CAQ-E values: 0.001 to 1000
            assert 0.001 < caq_e < 10000.0


class TestDeterminism:
    """Test deterministic behavior with fixed seeds."""

    def test_compression_deterministic(self):
        """Test that compression with same seed produces same results."""
        gradient = np.random.randn(50, 50).astype(np.float32)

        from experiments.run_energy_benchmark import compress_gradient_baseline

        # Run twice with same input
        compressed1 = compress_gradient_baseline(gradient)
        compressed2 = compress_gradient_baseline(gradient)

        # Should produce identical results
        assert len(compressed1) == len(compressed2)
        assert compressed1 == compressed2

    def test_seeded_randomness(self):
        """Test that random seed produces reproducible results."""
        np.random.seed(42)
        data1 = np.random.randn(100)

        np.random.seed(42)
        data2 = np.random.randn(100)

        # Should be identical
        assert np.allclose(data1, data2)


class TestNoNaNInResults:
    """Test that benchmark results never contain NaN or Inf."""

    def test_compute_caqe_no_nan(self):
        """Test that CAQ-E computation never produces NaN."""
        test_cases = [
            (2.0, 1.0, 10.0),
            (1.5, 0.0, 0.0),  # Edge case
            (10.0, 1e-9, 1e-9),  # Very small
            (0.5, 100.0, 1000.0),  # Poor compression
        ]

        for ratio, time, energy in test_cases:
            metrics = compute_caq_and_caqe(ratio, time, energy)

            # Check all metrics are finite
            assert np.isfinite(metrics["caq"])
            assert np.isfinite(metrics["caq_e"])
            assert np.isfinite(metrics["avg_power_watts"])
            assert np.isfinite(metrics["energy_efficiency"])

    def test_variance_computation_no_nan(self):
        """Test that variance computation never produces NaN."""
        from metrics.caq_energy_metric import compute_energy_variance

        test_cases = [
            [10.0, 12.0, 11.0, 13.0, 9.0],  # Normal
            [10.0],  # Single value
            [10.0, 10.0, 10.0],  # Identical values
            [0.0, 0.0, 0.0],  # All zeros
        ]

        for values in test_cases:
            variance = compute_energy_variance(values)

            assert np.isfinite(variance)
            assert variance >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

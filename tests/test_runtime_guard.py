"""
Unit Tests for RuntimeGuard (Phase H.5.1)

Tests runtime guardrails including:
- Finite metrics validation
- Variance gate (IQR/median ≤ 25%)
- Sanity range checks
- Rollback trigger detection
- Complete run validation

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Phase: H.5.1 - Runtime Guardrails and Variance Gate for CAQ-E Stability
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from energy.runtime_guard import (
    RuntimeGuard,
    validate_run,
    compute_variance_statistics,
)


class TestFiniteMetricsCheck:
    """Test 1-3: Finite metrics validation"""

    def test_finite_metrics_all_valid(self):
        """Test 1: All metrics are finite and valid"""
        guard = RuntimeGuard()
        metrics = {
            "compression_ratio": 2.5,
            "cpu_seconds": 0.123,
            "energy_joules": 5.67,
            "caq": 1.8,
            "caq_e": 0.35,
        }

        is_valid, error_msg = guard.check_finite_metrics(metrics)
        assert is_valid
        assert error_msg is None

    def test_finite_metrics_nan_detected(self):
        """Test 2: NaN values are detected"""
        guard = RuntimeGuard()
        metrics = {
            "compression_ratio": 2.5,
            "cpu_seconds": np.nan,
            "energy_joules": 5.67,
        }

        is_valid, error_msg = guard.check_finite_metrics(metrics)
        assert not is_valid
        assert "Non-finite" in error_msg
        assert "cpu_seconds" in error_msg

    def test_finite_metrics_inf_detected(self):
        """Test 3: Inf values are detected"""
        guard = RuntimeGuard()
        metrics = {
            "compression_ratio": np.inf,
            "energy_joules": 5.67,
        }

        is_valid, error_msg = guard.check_finite_metrics(metrics)
        assert not is_valid
        assert "Non-finite" in error_msg
        assert "compression_ratio" in error_msg


class TestVarianceGate:
    """Test 4-7: Variance gate (IQR/median ≤ 25%)"""

    def test_variance_gate_pass_low_variance(self):
        """Test 4: Low variance passes gate"""
        guard = RuntimeGuard()
        values = [1.0, 1.05, 0.98, 1.02, 1.01]  # ~5% variance

        passes, diagnostics = guard.check_variance_gate(values)

        assert passes
        assert diagnostics["variance_percent"] < 25.0
        assert diagnostics["passes"]

    def test_variance_gate_fail_high_variance(self):
        """Test 5: High variance fails gate"""
        guard = RuntimeGuard()
        values = [1.0, 2.0, 0.5, 1.8, 0.6]  # >25% variance

        passes, diagnostics = guard.check_variance_gate(values)

        assert not passes
        assert diagnostics["variance_percent"] > 25.0
        assert not diagnostics["passes"]

    def test_variance_gate_boundary_case(self):
        """Test 6: Exactly at 25% threshold"""
        guard = RuntimeGuard()
        # Construct values with IQR/median = 25%
        # For median = 1.0, we need IQR = 0.25
        # With 5 values: [q25, ..., median, ..., q75]
        median = 1.0
        iqr = 0.25
        q25 = median - iqr/2  # 0.875
        q75 = median + iqr/2  # 1.125

        # Create 5 values with these quartiles
        values = [q25 - 0.05, q25, median, q75, q75 + 0.05]

        passes, diagnostics = guard.check_variance_gate(values, threshold_percent=25.0)

        # Variance should be very close to 25%
        assert diagnostics["variance_percent"] <= 25.0
        assert diagnostics["passes"]

    def test_variance_gate_insufficient_samples(self):
        """Test 7: Single sample passes by default"""
        guard = RuntimeGuard()
        values = [1.0]

        passes, diagnostics = guard.check_variance_gate(values)

        assert passes
        assert diagnostics["reason"] == "insufficient_samples"


class TestSanityRangeChecks:
    """Test 8-10: Sanity range validation"""

    def test_sanity_range_all_valid(self):
        """Test 8: All values within valid ranges"""
        guard = RuntimeGuard()

        is_valid, violations = guard.check_sanity_range(
            compression_ratio=2.5,
            cpu_seconds=0.5,
            energy_joules=10.0
        )

        assert is_valid
        assert len(violations) == 0

    def test_sanity_range_invalid_compression_ratio(self):
        """Test 9: Invalid compression ratio detected"""
        guard = RuntimeGuard()

        # Test too low
        is_valid, violations = guard.check_sanity_range(
            compression_ratio=1e-10,  # Below MIN_COMPRESSION_RATIO
            cpu_seconds=0.5,
            energy_joules=10.0
        )

        assert not is_valid
        assert len(violations) > 0
        assert "Compression ratio" in violations[0]

        # Test too high
        is_valid, violations = guard.check_sanity_range(
            compression_ratio=1e5,  # Above MAX_COMPRESSION_RATIO
            cpu_seconds=0.5,
            energy_joules=10.0
        )

        assert not is_valid
        assert "Compression ratio" in violations[0]

    def test_sanity_range_invalid_energy(self):
        """Test 10: Invalid energy detected"""
        guard = RuntimeGuard()

        is_valid, violations = guard.check_sanity_range(
            compression_ratio=2.5,
            cpu_seconds=0.5,
            energy_joules=2e6  # Above MAX_ENERGY_JOULES
        )

        assert not is_valid
        assert len(violations) > 0
        assert "Energy" in violations[0]


class TestRollbackTrigger:
    """Test 11-13: Rollback trigger detection"""

    def test_rollback_no_checkpoint(self):
        """Test 11: No rollback when checkpoint not set"""
        guard = RuntimeGuard()

        should_rollback, diagnostics = guard.check_rollback_trigger(1.5)

        assert not should_rollback
        assert diagnostics["reason"] == "no_checkpoint"

    def test_rollback_triggered_on_drop(self):
        """Test 12: Rollback triggered on >5% drop"""
        guard = RuntimeGuard(enable_rollback=True)

        # Create checkpoint at 1.0
        guard.create_checkpoint(1.0, metadata={"test": "checkpoint"})

        # Current value dropped to 0.9 (10% drop > 5% threshold)
        should_rollback, diagnostics = guard.check_rollback_trigger(0.9)

        assert should_rollback
        assert diagnostics["drop_percent"] == pytest.approx(10.0, abs=0.1)
        assert diagnostics["should_rollback"]

    def test_rollback_not_triggered_small_drop(self):
        """Test 13: Rollback not triggered on <5% drop"""
        guard = RuntimeGuard(enable_rollback=True)

        # Create checkpoint at 1.0
        guard.create_checkpoint(1.0)

        # Current value dropped to 0.97 (3% drop < 5% threshold)
        should_rollback, diagnostics = guard.check_rollback_trigger(0.97)

        assert not should_rollback
        assert diagnostics["drop_percent"] < 5.0


class TestCompleteRunValidation:
    """Test 14-15: Complete run validation"""

    def test_validate_run_all_pass(self):
        """Test 14: Valid run passes all guardrails"""
        guard = RuntimeGuard()

        report = {
            "compression_ratio": 2.5,
            "cpu_seconds": 0.5,
            "energy_joules": 10.0,
            "caq": 1.67,
            "caq_e": 0.238,
        }

        is_valid, guard_status = guard.validate_run(report)

        assert is_valid
        assert guard_status["finite"]
        assert guard_status["sanity_pass"]

    def test_validate_run_fails_on_nan(self):
        """Test 15: Invalid run fails guardrails"""
        guard = RuntimeGuard()

        report = {
            "compression_ratio": 2.5,
            "cpu_seconds": np.nan,  # Invalid
            "energy_joules": 10.0,
            "caq": 1.67,
        }

        is_valid, guard_status = guard.validate_run(report)

        assert not is_valid
        assert not guard_status["finite"]
        assert "finite_error" in guard_status["details"]


class TestVarianceStatistics:
    """Additional tests for variance statistics computation"""

    def test_compute_variance_statistics_normal(self):
        """Test variance statistics with normal data"""
        values = [1.0, 1.1, 0.9, 1.05, 0.95]

        stats = compute_variance_statistics(values)

        assert stats["count"] == 5
        assert stats["median"] == pytest.approx(1.0, abs=0.05)
        assert stats["iqr"] > 0
        assert stats["variance_percent"] < 25.0

    def test_compute_variance_statistics_single_value(self):
        """Test variance statistics with single value"""
        values = [5.0]

        stats = compute_variance_statistics(values)

        assert stats["count"] == 1
        assert stats["median"] == 5.0
        assert stats["iqr"] == 0.0
        assert stats["variance_percent"] == 0.0

    def test_compute_variance_statistics_zero_variance(self):
        """Test variance statistics with identical values"""
        values = [2.0, 2.0, 2.0, 2.0]

        stats = compute_variance_statistics(values)

        assert stats["variance_percent"] == 0.0
        assert stats["iqr"] == 0.0


class TestConvenienceFunction:
    """Test convenience function for run validation"""

    def test_validate_run_convenience_function(self):
        """Test validate_run convenience function"""
        report = {
            "compression_ratio": 2.5,
            "cpu_seconds": 0.5,
            "energy_joules": 10.0,
        }

        is_valid = validate_run(report)
        assert is_valid

    def test_validate_run_strict_mode(self):
        """Test validate_run with strict mode"""
        report = {
            "compression_ratio": 2.5,
            "cpu_seconds": np.nan,
            "energy_joules": 10.0,
        }

        # Strict mode doesn't change validation, just logging
        is_valid = validate_run(report, strict=True)
        assert not is_valid


class TestNegativeDelta:
    """Test negative delta (regression) detection"""

    def test_negative_delta_detected(self):
        """Test negative delta detection"""
        guard = RuntimeGuard()

        is_positive, warning = guard.check_negative_delta(-15.5)

        assert not is_positive
        assert "Negative CAQ-E delta" in warning

    def test_positive_delta_ok(self):
        """Test positive delta passes"""
        guard = RuntimeGuard()

        is_positive, warning = guard.check_negative_delta(25.0)

        assert is_positive
        assert warning is None


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_median_variance_gate(self):
        """Test variance gate with zero median"""
        guard = RuntimeGuard()

        values = [0.0, 0.0, 0.0]

        passes, diagnostics = guard.check_variance_gate(values)

        assert not passes
        assert diagnostics["reason"] == "zero_median"

    def test_non_finite_variance_gate(self):
        """Test variance gate with non-finite values"""
        guard = RuntimeGuard()

        values = [1.0, np.nan, 1.5]

        passes, diagnostics = guard.check_variance_gate(values)

        assert not passes
        assert diagnostics["reason"] == "non_finite_values"

    def test_checkpoint_metadata(self):
        """Test checkpoint with metadata"""
        guard = RuntimeGuard()

        guard.create_checkpoint(1.5, metadata={"version": "1.0", "dataset": "test"})

        assert guard.checkpoint is not None
        assert guard.checkpoint["median_caqe"] == 1.5
        assert guard.checkpoint["metadata"]["version"] == "1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

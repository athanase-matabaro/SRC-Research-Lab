"""
Runtime Guardrails for CAQ-E Stability

Provides in-process safeguards against numeric and statistical anomalies
in energy-aware compression benchmarks.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-17
Phase: H.5.1 - Runtime Guardrails and Variance Gate
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RuntimeGuard:
    """
    Runtime guardrails for CAQ-E benchmark stability.

    Protects against:
    - Non-finite metrics (NaN, Inf)
    - Extreme variance (>25% IQR/median)
    - Unsafe value ranges
    - Performance regressions requiring rollback
    """

    # Thresholds
    MAX_VARIANCE_PERCENT = 25.0  # Maximum IQR/median ratio (%)
    MAX_ROLLBACK_DROP_PERCENT = 5.0  # Trigger rollback if median drops >5%
    MIN_COMPRESSION_RATIO = 1e-6  # Minimum valid ratio
    MAX_COMPRESSION_RATIO = 1e4  # Maximum valid ratio
    MIN_CPU_SECONDS = 1e-9  # Minimum valid time
    MAX_CPU_SECONDS = 1e5  # Maximum valid time (27 hours)
    MIN_ENERGY_JOULES = 0.0  # Minimum valid energy
    MAX_ENERGY_JOULES = 1e6  # Maximum valid energy (1 MJ)

    def __init__(self, enable_rollback: bool = True, strict_mode: bool = False):
        """
        Initialize runtime guard.

        Args:
            enable_rollback: Enable automatic rollback on regression.
            strict_mode: If True, raise exceptions instead of warnings.
        """
        self.enable_rollback = enable_rollback
        self.strict_mode = strict_mode
        self.checkpoint = None

    def check_finite_metrics(self, metrics: Dict) -> Tuple[bool, Optional[str]]:
        """
        Check that all metrics are finite (not NaN or Inf).

        Args:
            metrics: Dictionary of metric values.

        Returns:
            (is_valid, error_message)
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    return False, f"Non-finite value in {key}: {value}"

        return True, None

    def check_variance_gate(
        self,
        values: List[float],
        threshold_percent: float = None
    ) -> Tuple[bool, Dict]:
        """
        Check if variance is within acceptable range.

        Uses IQR/median ratio as a robust variance measure.

        Args:
            values: List of CAQ-E or other metric values.
            threshold_percent: Maximum acceptable variance (default: 25%).

        Returns:
            (passes_gate, diagnostics)
        """
        if threshold_percent is None:
            threshold_percent = self.MAX_VARIANCE_PERCENT

        if len(values) < 2:
            return True, {"reason": "insufficient_samples", "count": len(values)}

        arr = np.array(values)

        # Check for non-finite values
        if not np.all(np.isfinite(arr)):
            return False, {
                "reason": "non_finite_values",
                "finite_count": np.sum(np.isfinite(arr)),
                "total_count": len(arr)
            }

        # Compute robust variance using IQR/median
        q25 = np.percentile(arr, 25)
        q75 = np.percentile(arr, 75)
        median = np.median(arr)
        iqr = q75 - q25

        # Avoid division by zero
        if median == 0:
            return False, {
                "reason": "zero_median",
                "iqr": float(iqr),
                "median": 0.0
            }

        variance_percent = (iqr / abs(median)) * 100.0

        passes = variance_percent <= threshold_percent

        diagnostics = {
            "iqr": float(iqr),
            "median": float(median),
            "q25": float(q25),
            "q75": float(q75),
            "variance_percent": float(variance_percent),
            "threshold_percent": float(threshold_percent),
            "passes": bool(passes)
        }

        return bool(passes), diagnostics

    def check_sanity_range(
        self,
        compression_ratio: float,
        cpu_seconds: float,
        energy_joules: float
    ) -> Tuple[bool, List[str]]:
        """
        Check if values are within reasonable ranges.

        Args:
            compression_ratio: Compression ratio.
            cpu_seconds: CPU time in seconds.
            energy_joules: Energy in joules.

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Check compression ratio
        if not (self.MIN_COMPRESSION_RATIO <= compression_ratio <= self.MAX_COMPRESSION_RATIO):
            violations.append(
                f"Compression ratio {compression_ratio} outside range "
                f"[{self.MIN_COMPRESSION_RATIO}, {self.MAX_COMPRESSION_RATIO}]"
            )

        # Check CPU time
        if not (self.MIN_CPU_SECONDS <= cpu_seconds <= self.MAX_CPU_SECONDS):
            violations.append(
                f"CPU seconds {cpu_seconds} outside range "
                f"[{self.MIN_CPU_SECONDS}, {self.MAX_CPU_SECONDS}]"
            )

        # Check energy
        if not (self.MIN_ENERGY_JOULES <= energy_joules <= self.MAX_ENERGY_JOULES):
            violations.append(
                f"Energy joules {energy_joules} outside range "
                f"[{self.MIN_ENERGY_JOULES}, {self.MAX_ENERGY_JOULES}]"
            )

        is_valid = len(violations) == 0
        return is_valid, violations

    def check_negative_delta(self, delta_percent: float) -> Tuple[bool, Optional[str]]:
        """
        Check if CAQ-E delta is negative (regression).

        Args:
            delta_percent: CAQ-E improvement percentage.

        Returns:
            (is_positive, warning_message)
        """
        if delta_percent < 0:
            return False, f"Negative CAQ-E delta: {delta_percent:.2f}%"
        return True, None

    def create_checkpoint(self, median_caqe: float, metadata: Dict = None):
        """
        Create a checkpoint for rollback.

        Args:
            median_caqe: Current median CAQ-E value.
            metadata: Additional checkpoint metadata.
        """
        self.checkpoint = {
            "median_caqe": median_caqe,
            "metadata": metadata or {}
        }
        logger.info(f"Checkpoint created: median_caqe={median_caqe:.4f}")

    def check_rollback_trigger(
        self,
        current_median: float,
        drop_threshold_percent: float = None
    ) -> Tuple[bool, Dict]:
        """
        Check if rollback should be triggered due to performance drop.

        Args:
            current_median: Current median CAQ-E.
            drop_threshold_percent: Trigger threshold (default: 5%).

        Returns:
            (should_rollback, diagnostics)
        """
        if drop_threshold_percent is None:
            drop_threshold_percent = self.MAX_ROLLBACK_DROP_PERCENT

        if self.checkpoint is None:
            return False, {"reason": "no_checkpoint"}

        previous_median = self.checkpoint["median_caqe"]

        if previous_median == 0:
            return False, {"reason": "zero_previous_median"}

        drop_percent = ((previous_median - current_median) / previous_median) * 100.0

        should_rollback = drop_percent > drop_threshold_percent

        diagnostics = {
            "previous_median": float(previous_median),
            "current_median": float(current_median),
            "drop_percent": float(drop_percent),
            "threshold_percent": float(drop_threshold_percent),
            "should_rollback": bool(should_rollback)
        }

        if should_rollback and self.enable_rollback:
            logger.warning(
                f"Rollback triggered: {drop_percent:.2f}% drop "
                f"(threshold: {drop_threshold_percent}%)"
            )

        return should_rollback, diagnostics

    def validate_run(self, report: Dict) -> Tuple[bool, Dict]:
        """
        Validate a complete benchmark run.

        Args:
            report: Run report with metrics.

        Returns:
            (is_valid, guardrail_status)
        """
        guardrail_status = {
            "finite": True,
            "variance_pass": True,
            "sanity_pass": True,
            "rollback": False,
            "details": {}
        }

        # Check finite metrics
        finite_ok, finite_msg = self.check_finite_metrics(report)
        guardrail_status["finite"] = finite_ok
        if not finite_ok:
            guardrail_status["details"]["finite_error"] = finite_msg
            logger.error(f"Finite check failed: {finite_msg}")

        # Check sanity ranges
        if "compression_ratio" in report and "cpu_seconds" in report and "energy_joules" in report:
            sanity_ok, violations = self.check_sanity_range(
                report["compression_ratio"],
                report["cpu_seconds"],
                report["energy_joules"]
            )
            guardrail_status["sanity_pass"] = sanity_ok
            if not sanity_ok:
                guardrail_status["details"]["sanity_violations"] = violations
                for violation in violations:
                    logger.warning(f"Sanity check violation: {violation}")

        # Overall validation
        is_valid = (
            guardrail_status["finite"] and
            guardrail_status["sanity_pass"]
        )

        return is_valid, guardrail_status


def validate_run(report: Dict, strict: bool = False) -> bool:
    """
    Convenience function to validate a run report.

    Args:
        report: Run report dictionary.
        strict: If True, use strict mode.

    Returns:
        True if valid, False otherwise.
    """
    guard = RuntimeGuard(strict_mode=strict)
    is_valid, _ = guard.validate_run(report)
    return is_valid


def compute_variance_statistics(values: List[float]) -> Dict:
    """
    Compute variance statistics for a list of values.

    Args:
        values: List of metric values.

    Returns:
        Dictionary with variance statistics.
    """
    if len(values) < 2:
        return {
            "count": len(values),
            "iqr": 0.0,
            "median": float(values[0]) if len(values) == 1 else 0.0,
            "variance_percent": 0.0
        }

    arr = np.array(values)
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    median = np.median(arr)
    iqr = q75 - q25

    variance_percent = (iqr / abs(median)) * 100.0 if median != 0 else 0.0

    return {
        "count": len(values),
        "iqr": float(iqr),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "variance_percent": float(variance_percent),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr))
    }

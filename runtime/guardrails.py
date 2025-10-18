"""
Phase B1 Runtime Guardrails: Variance Gate and Rollback System

This module implements runtime monitoring and stability detection for compression
benchmarks, with config-aware thresholds informed by Phase C2 cross-platform analysis.

Key Features:
- Rolling window variance monitoring
- Config-aware drift detection
- Automatic rollback on instability
- File-based state persistence
- Energy-CAQ coherence validation

Design informed by Phase C2 findings:
- Baseline variance: 47% (environmental + scheduling)
- Cross-config variance: 44% (hardware-induced)
- Energy-CAQ correlation: -0.92 (strong coherence expected)
"""

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np


@dataclass
class BaselineStats:
    """Baseline statistics for drift computation."""
    mean_caq_e: float
    std_caq_e: float
    mean_energy: float
    std_energy: float
    config_name: str = "baseline"
    num_samples: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'BaselineStats':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GuardrailState:
    """Current state of the guardrail system."""
    is_stable: bool
    current_variance_percent: float
    current_drift_percent: float
    window_size: int
    total_updates: int
    last_rollback_time: Optional[float] = None
    consecutive_stable_checks: int = 0
    rollback_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'GuardrailState':
        """Create from dictionary."""
        return cls(**data)


class GuardrailManager:
    """
    Runtime variance gate with config-aware thresholds.

    Monitors CAQ-E and energy metrics in a rolling window, triggers rollback
    when variance or drift exceeds thresholds, and resumes when stability returns.

    Thresholds calibrated from Phase C2 empirical data:
    - Intra-config variance: 47-74% observed (use 75% as flag threshold)
    - Cross-config drift: up to 67% observed (use 70% as flag threshold)
    - Energy-CAQ correlation: -0.92 expected (flag if < -0.95 or > -0.85)

    Args:
        baseline_stats: Reference statistics for drift computation
        window: Rolling window size (default=20, balances responsiveness vs noise)
        drift_threshold: Max acceptable drift from baseline (default=0.15 = 15%)
        variance_threshold: Max acceptable intra-window variance (default=0.75 = 75%)
        energy_correlation_range: Expected energy-CAQ correlation range (default=(-0.95, -0.85))
        state_file: Path to persistence file (default=runtime/guardrail_state.json)
    """

    def __init__(
        self,
        baseline_stats: BaselineStats,
        window: int = 20,
        drift_threshold: float = 0.15,
        variance_threshold: float = 0.75,
        energy_correlation_range: Tuple[float, float] = (-0.95, -0.85),
        state_file: Path = Path("runtime/guardrail_state.json")
    ):
        # Validate inputs
        if not math.isfinite(baseline_stats.mean_caq_e) or baseline_stats.mean_caq_e <= 0:
            raise ValueError(f"Invalid baseline CAQ-E mean: {baseline_stats.mean_caq_e}")
        if not math.isfinite(baseline_stats.std_caq_e) or baseline_stats.std_caq_e < 0:
            raise ValueError(f"Invalid baseline CAQ-E std: {baseline_stats.std_caq_e}")
        if not math.isfinite(baseline_stats.mean_energy) or baseline_stats.mean_energy <= 0:
            raise ValueError(f"Invalid baseline energy mean: {baseline_stats.mean_energy}")
        if window < 2:
            raise ValueError(f"Window must be >= 2, got {window}")
        if not (0 < drift_threshold < 1):
            raise ValueError(f"Drift threshold must be in (0, 1), got {drift_threshold}")
        if not (0 < variance_threshold < 2):
            raise ValueError(f"Variance threshold must be in (0, 2), got {variance_threshold}")

        self.baseline = baseline_stats
        self.window = window
        self.drift_threshold = drift_threshold
        self.variance_threshold = variance_threshold
        self.energy_correlation_range = energy_correlation_range
        self.state_file = Path(state_file)

        # Rolling window buffers
        self.caq_e_buffer: List[float] = []
        self.energy_buffer: List[float] = []
        self.timestamps: List[float] = []

        # State tracking
        self.state = GuardrailState(
            is_stable=True,
            current_variance_percent=0.0,
            current_drift_percent=0.0,
            window_size=window,
            total_updates=0
        )

        # Ensure state directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def update_metrics(self, caq_e: float, energy: float) -> Dict[str, any]:
        """
        Update rolling window with new metrics.

        Args:
            caq_e: Current CAQ-E value
            energy: Current energy consumption (Joules)

        Returns:
            Dict with update status and computed metrics

        Raises:
            ValueError: If inputs are NaN or infinite
        """
        # Validate inputs
        if not math.isfinite(caq_e):
            raise ValueError(f"Invalid CAQ-E value: {caq_e} (must be finite)")
        if not math.isfinite(energy):
            raise ValueError(f"Invalid energy value: {energy} (must be finite)")
        if caq_e <= 0:
            raise ValueError(f"CAQ-E must be positive, got {caq_e}")
        if energy <= 0:
            raise ValueError(f"Energy must be positive, got {energy}")

        # Add to buffers
        self.caq_e_buffer.append(caq_e)
        self.energy_buffer.append(energy)
        self.timestamps.append(time.time())

        # Maintain window size
        if len(self.caq_e_buffer) > self.window:
            self.caq_e_buffer.pop(0)
            self.energy_buffer.pop(0)
            self.timestamps.pop(0)

        self.state.total_updates += 1

        # Compute metrics (only if we have enough samples)
        metrics = {
            "caq_e": caq_e,
            "energy": energy,
            "timestamp": self.timestamps[-1],
            "window_fill": len(self.caq_e_buffer),
            "window_capacity": self.window
        }

        if len(self.caq_e_buffer) >= 2:
            # Compute variance using IQR/median ratio (robust to outliers)
            variance_stats = self._compute_variance_stats(self.caq_e_buffer)
            metrics["variance_percent"] = variance_stats["variance_percent"]
            metrics["iqr_median_ratio"] = variance_stats["iqr_median_ratio"]

            # Compute drift from baseline
            drift_stats = self._compute_drift_stats(self.caq_e_buffer)
            metrics["drift_percent"] = drift_stats["drift_percent"]
            metrics["window_mean"] = drift_stats["window_mean"]

            # Update state
            self.state.current_variance_percent = variance_stats["variance_percent"]
            self.state.current_drift_percent = drift_stats["drift_percent"]

            # Compute energy-CAQ correlation (if window is large enough)
            if len(self.caq_e_buffer) >= 3:
                correlation = self._compute_energy_correlation()
                metrics["energy_caq_correlation"] = correlation

        return metrics

    def check_stability(self) -> Dict[str, any]:
        """
        Check if current metrics are within acceptable thresholds.

        Returns:
            Dict with stability status and violation details
        """
        if len(self.caq_e_buffer) < 2:
            # Not enough data yet
            return {
                "stable": True,
                "reason": "insufficient_data",
                "window_fill": len(self.caq_e_buffer),
                "checks_performed": []
            }

        violations = []
        checks = []

        # Check 1: Variance gate (IQR/median ratio)
        variance_percent = self.state.current_variance_percent
        variance_check = {
            "name": "variance_gate",
            "value": variance_percent,
            "threshold": self.variance_threshold * 100,
            "passed": variance_percent <= self.variance_threshold * 100
        }
        checks.append(variance_check)
        if not variance_check["passed"]:
            violations.append(f"Variance {variance_percent:.1f}% exceeds {self.variance_threshold*100:.0f}%")

        # Check 2: Drift from baseline
        drift_percent = self.state.current_drift_percent
        drift_check = {
            "name": "drift_gate",
            "value": drift_percent,
            "threshold": self.drift_threshold * 100,
            "passed": abs(drift_percent) <= self.drift_threshold * 100
        }
        checks.append(drift_check)
        if not drift_check["passed"]:
            violations.append(f"Drift {drift_percent:.1f}% exceeds ±{self.drift_threshold*100:.0f}%")

        # Check 3: Energy-CAQ correlation (if we have enough samples)
        if len(self.caq_e_buffer) >= 3:
            correlation = self._compute_energy_correlation()
            correlation_check = {
                "name": "energy_coherence",
                "value": correlation,
                "threshold_range": self.energy_correlation_range,
                "passed": self.energy_correlation_range[0] <= correlation <= self.energy_correlation_range[1]
            }
            checks.append(correlation_check)
            if not correlation_check["passed"]:
                violations.append(
                    f"Energy-CAQ correlation {correlation:.3f} outside expected range "
                    f"[{self.energy_correlation_range[0]:.3f}, {self.energy_correlation_range[1]:.3f}]"
                )

        # Check 4: Finite value sanity checks
        for val in self.caq_e_buffer[-5:]:  # Check last 5 values
            if not math.isfinite(val):
                violations.append(f"Non-finite CAQ-E value detected: {val}")
                checks.append({
                    "name": "finite_values",
                    "value": val,
                    "passed": False
                })
                break
        else:
            checks.append({"name": "finite_values", "passed": True})

        # Determine overall stability
        is_stable = len(violations) == 0

        # Update consecutive stable counter
        if is_stable:
            self.state.consecutive_stable_checks += 1
        else:
            self.state.consecutive_stable_checks = 0

        # Update state
        previous_stable = self.state.is_stable
        self.state.is_stable = is_stable

        return {
            "stable": is_stable,
            "was_stable": previous_stable,
            "violations": violations,
            "checks_performed": checks,
            "consecutive_stable": self.state.consecutive_stable_checks,
            "window_fill": len(self.caq_e_buffer)
        }

    def trigger_rollback(self, reason: str = "instability_detected") -> Dict[str, any]:
        """
        Trigger rollback and persist state.

        Args:
            reason: Human-readable reason for rollback

        Returns:
            Dict with rollback status and state file path
        """
        self.state.is_stable = False
        self.state.last_rollback_time = time.time()
        self.state.rollback_count += 1
        self.state.consecutive_stable_checks = 0

        # Persist state to disk
        state_data = {
            "timestamp": time.time(),
            "reason": reason,
            "baseline": self.baseline.to_dict(),
            "state": self.state.to_dict(),
            "window_config": {
                "size": self.window,
                "drift_threshold": self.drift_threshold,
                "variance_threshold": self.variance_threshold,
                "energy_correlation_range": self.energy_correlation_range
            },
            "current_metrics": {
                "variance_percent": self.state.current_variance_percent,
                "drift_percent": self.state.current_drift_percent,
                "window_fill": len(self.caq_e_buffer)
            }
        }

        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

        return {
            "rollback_triggered": True,
            "reason": reason,
            "state_file": str(self.state_file),
            "rollback_count": self.state.rollback_count,
            "timestamp": state_data["timestamp"]
        }

    def resume_if_stable(self, required_consecutive: int = 5) -> Dict[str, any]:
        """
        Resume from rollback if stability has been maintained.

        Args:
            required_consecutive: Number of consecutive stable checks required (default=5)

        Returns:
            Dict with resume status and stability metrics
        """
        can_resume = (
            not self.state.is_stable and  # Currently in rollback
            self.state.consecutive_stable_checks >= required_consecutive
        )

        if can_resume:
            self.state.is_stable = True

            # Update state file
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)

                state_data["resumed_at"] = time.time()
                state_data["consecutive_stable_at_resume"] = self.state.consecutive_stable_checks
                state_data["state"] = self.state.to_dict()

                with open(self.state_file, 'w') as f:
                    json.dump(state_data, f, indent=2)

            return {
                "resumed": True,
                "consecutive_stable": self.state.consecutive_stable_checks,
                "required": required_consecutive,
                "timestamp": time.time()
            }
        else:
            return {
                "resumed": False,
                "consecutive_stable": self.state.consecutive_stable_checks,
                "required": required_consecutive,
                "reason": "stable" if self.state.is_stable else f"insufficient_consecutive ({self.state.consecutive_stable_checks}/{required_consecutive})"
            }

    def _compute_variance_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Compute robust variance statistics using IQR/median ratio.

        This metric is more robust to outliers than standard deviation.
        Phase C2 baseline showed 47% IQR/median ratio, so we use 75% as threshold.
        """
        if len(values) < 2:
            return {"variance_percent": 0.0, "iqr_median_ratio": 0.0}

        arr = np.array(values)
        median = np.median(arr)

        if median == 0:
            # Avoid division by zero
            return {"variance_percent": 0.0, "iqr_median_ratio": 0.0}

        q25 = np.percentile(arr, 25)
        q75 = np.percentile(arr, 75)
        iqr = q75 - q25

        iqr_median_ratio = (iqr / median) * 100

        return {
            "variance_percent": iqr_median_ratio,
            "iqr_median_ratio": iqr_median_ratio,
            "median": median,
            "iqr": iqr,
            "q25": q25,
            "q75": q75
        }

    def _compute_drift_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Compute drift from baseline mean.

        Drift index = (window_mean - baseline_mean) / baseline_mean * 100
        Phase C2 showed up to 67% drift across configs, so we use 15% as default threshold
        for intra-config monitoring (more conservative).
        """
        if len(values) < 1:
            return {"drift_percent": 0.0, "window_mean": 0.0}

        window_mean = np.mean(values)

        if self.baseline.mean_caq_e == 0:
            return {"drift_percent": 0.0, "window_mean": window_mean}

        drift_percent = ((window_mean - self.baseline.mean_caq_e) / self.baseline.mean_caq_e) * 100

        return {
            "drift_percent": drift_percent,
            "window_mean": window_mean,
            "baseline_mean": self.baseline.mean_caq_e
        }

    def _compute_energy_correlation(self) -> float:
        """
        Compute Pearson correlation between energy and CAQ-E.

        Phase C2 showed -0.92 correlation (strong negative: higher energy → lower CAQ-E).
        We expect correlations in range [-0.95, -0.85] for coherent behavior.
        """
        if len(self.energy_buffer) < 3 or len(self.caq_e_buffer) < 3:
            return 0.0  # Not enough data

        # Use numpy for robust correlation computation
        try:
            correlation = np.corrcoef(self.energy_buffer, self.caq_e_buffer)[0, 1]

            # Handle edge case where correlation is NaN (constant values)
            if not math.isfinite(correlation):
                return 0.0

            return correlation
        except Exception:
            return 0.0

    def get_state(self) -> GuardrailState:
        """Get current guardrail state."""
        return self.state

    def get_baseline(self) -> BaselineStats:
        """Get baseline statistics."""
        return self.baseline

    def get_window_metrics(self) -> Dict[str, any]:
        """
        Get current window metrics for monitoring.

        Returns:
            Dict with all current window statistics
        """
        if len(self.caq_e_buffer) < 2:
            return {
                "window_fill": len(self.caq_e_buffer),
                "window_size": self.window,
                "ready": False
            }

        variance_stats = self._compute_variance_stats(self.caq_e_buffer)
        drift_stats = self._compute_drift_stats(self.caq_e_buffer)

        metrics = {
            "window_fill": len(self.caq_e_buffer),
            "window_size": self.window,
            "ready": True,
            "caq_e": {
                "mean": np.mean(self.caq_e_buffer),
                "median": np.median(self.caq_e_buffer),
                "std": np.std(self.caq_e_buffer),
                "min": np.min(self.caq_e_buffer),
                "max": np.max(self.caq_e_buffer)
            },
            "variance": variance_stats,
            "drift": drift_stats
        }

        if len(self.caq_e_buffer) >= 3:
            metrics["energy_caq_correlation"] = self._compute_energy_correlation()

        return metrics

    def clear_window(self):
        """Clear rolling window buffers (useful for testing or reset)."""
        self.caq_e_buffer.clear()
        self.energy_buffer.clear()
        self.timestamps.clear()
        self.state.consecutive_stable_checks = 0

    @classmethod
    def load_from_file(cls, state_file: Path) -> 'GuardrailManager':
        """
        Load GuardrailManager from persisted state file.

        Args:
            state_file: Path to state JSON file

        Returns:
            GuardrailManager instance restored from file

        Raises:
            FileNotFoundError: If state file doesn't exist
            ValueError: If state file is invalid
        """
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        with open(state_file, 'r') as f:
            data = json.load(f)

        # Reconstruct baseline
        baseline = BaselineStats.from_dict(data["baseline"])

        # Reconstruct manager
        window_config = data["window_config"]
        manager = cls(
            baseline_stats=baseline,
            window=window_config["size"],
            drift_threshold=window_config["drift_threshold"],
            variance_threshold=window_config["variance_threshold"],
            energy_correlation_range=tuple(window_config["energy_correlation_range"]),
            state_file=state_file
        )

        # Restore state
        manager.state = GuardrailState.from_dict(data["state"])

        return manager

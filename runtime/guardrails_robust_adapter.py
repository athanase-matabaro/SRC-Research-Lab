#!/usr/bin/env python3
"""
Robust Aggregator Adapter for Runtime Guardrails (Phase D.1 Integration)

Provides a compatibility layer between RobustAggregator (Phase D.1) and
existing GuardrailManager (Phase B.1), enabling optional robust statistics
while preserving backward compatibility with mean-based thresholds.

Key Features:
- Configurable robust mode (auto, trimmed, winsor, mad, or disabled)
- Dual-tracking: stores both mean and robust statistics for traceability
- Backward compatible: defaults to mean-based aggregation
- Rollback semantics: preserves same parameters across rollbacks
- Performance: minimal overhead (<2x mean aggregator)

Design:
    GuardrailManager → RobustGuardrailAdapter → RobustAggregator
                    ↓
                 Stores both mean + robust values in state
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add metrics to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics.caq_robust import RobustAggregator


@dataclass
class RobustBaselineStats:
    """
    Extended baseline statistics including robust estimators.

    Backward compatible with BaselineStats - adds robust fields
    that can be ignored by legacy code.
    """
    # Standard (mean-based) statistics
    mean_caq_e: float
    std_caq_e: float
    mean_energy: float
    std_energy: float
    config_name: str = "baseline"
    num_samples: int = 0

    # Robust statistics (optional, populated when robust mode enabled)
    robust_caq_e: Optional[float] = None
    robust_energy: Optional[float] = None
    robust_std_caq_e: Optional[float] = None  # MAD-based
    robust_std_energy: Optional[float] = None
    robust_mode: Optional[str] = None  # auto, trimmed, winsor, mad, or None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'RobustBaselineStats':
        """Create from dictionary, handling legacy BaselineStats."""
        # Handle legacy format (no robust fields)
        if 'robust_caq_e' not in data:
            data['robust_caq_e'] = None
            data['robust_energy'] = None
            data['robust_std_caq_e'] = None
            data['robust_std_energy'] = None
            data['robust_mode'] = None

        return cls(**data)


class RobustGuardrailAdapter:
    """
    Adapter layer for integrating RobustAggregator with GuardrailManager.

    Responsibilities:
    - Maintains parallel aggregators for CAQ-E and energy metrics
    - Computes both mean and robust statistics
    - Provides compatibility methods for existing guardrail code
    - Manages rollback semantics (preserves aggregator state)

    Usage:
        # Create adapter with robust mode enabled
        adapter = RobustGuardrailAdapter(
            robust_mode='auto',
            window=20
        )

        # Update with new metrics
        adapter.update(caq_e=4.5, energy=12.3)

        # Get current statistics
        stats = adapter.get_current_stats()

        # Check which value to use for guardrail decisions
        active_value = adapter.get_active_caq_e()
    """

    def __init__(
        self,
        robust_mode: Optional[str] = None,
        window: int = 20,
        trim_pct: float = 0.1,
        winsor_pct: float = 0.05,
        seed: Optional[int] = None
    ):
        """
        Initialize robust aggregator adapter.

        Args:
            robust_mode: Robust estimator mode ('auto', 'trimmed', 'winsor', 'mad', or None for disabled)
            window: Rolling window size
            trim_pct: Trimming percentage for trimmed mean
            winsor_pct: Winsorizing percentage for winsorized mean
            seed: Random seed for deterministic behavior
        """
        self.robust_mode = robust_mode
        self.window = window
        self.enabled = (robust_mode is not None)

        # Create aggregators for CAQ-E
        if self.enabled:
            self.caq_e_robust = RobustAggregator(
                mode=robust_mode,
                window=window,
                trim_pct=trim_pct,
                winsor_pct=winsor_pct,
                seed=seed
            )
        else:
            self.caq_e_robust = None

        # Always maintain mean aggregator for backward compatibility
        self.caq_e_mean = RobustAggregator(
            mode='mean',
            window=window,
            seed=seed
        )

        # Create aggregators for energy
        if self.enabled:
            self.energy_robust = RobustAggregator(
                mode=robust_mode,
                window=window,
                trim_pct=trim_pct,
                winsor_pct=winsor_pct,
                seed=seed
            )
        else:
            self.energy_robust = None

        self.energy_mean = RobustAggregator(
            mode='mean',
            window=window,
            seed=seed
        )

        self.update_count = 0

    def update(self, caq_e: float, energy: float):
        """
        Update aggregators with new metric values.

        Args:
            caq_e: CAQ-E value
            energy: Energy value (joules)
        """
        # Update mean aggregators (always)
        self.caq_e_mean.update(caq_e)
        self.energy_mean.update(energy)

        # Update robust aggregators (if enabled)
        if self.enabled:
            self.caq_e_robust.update(caq_e)
            self.energy_robust.update(energy)

        self.update_count += 1

    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current statistics from all aggregators.

        Returns:
            Dictionary containing:
                - mean_caq_e, std_caq_e: Mean-based statistics
                - mean_energy, std_energy: Mean-based statistics
                - robust_caq_e, robust_std_caq_e: Robust statistics (if enabled)
                - robust_energy, robust_std_energy: Robust statistics (if enabled)
                - robust_mode: Active robust mode
                - n_samples: Number of updates
        """
        # Get mean statistics
        caq_e_summary = self.caq_e_mean.summary()
        energy_summary = self.energy_mean.summary()

        stats = {
            'mean_caq_e': caq_e_summary['mean'],
            'std_caq_e': np.sqrt(caq_e_summary['variance']),
            'mean_energy': energy_summary['mean'],
            'std_energy': np.sqrt(energy_summary['variance']),
            'n_samples': self.update_count,
            'robust_mode': self.robust_mode
        }

        # Add robust statistics if enabled
        if self.enabled:
            caq_e_robust_summary = self.caq_e_robust.summary()
            energy_robust_summary = self.energy_robust.summary()

            stats['robust_caq_e'] = caq_e_robust_summary['value']
            stats['robust_std_caq_e'] = np.sqrt(caq_e_robust_summary['variance'])
            stats['robust_energy'] = energy_robust_summary['value']
            stats['robust_std_energy'] = np.sqrt(energy_robust_summary['variance'])
            stats['selected_estimator_caq_e'] = caq_e_robust_summary['estimator']
            stats['selected_estimator_energy'] = energy_robust_summary['estimator']
        else:
            stats['robust_caq_e'] = None
            stats['robust_std_caq_e'] = None
            stats['robust_energy'] = None
            stats['robust_std_energy'] = None
            stats['selected_estimator_caq_e'] = None
            stats['selected_estimator_energy'] = None

        return stats

    def get_active_caq_e(self) -> float:
        """
        Get active CAQ-E value for guardrail decisions.

        Returns robust value if enabled, otherwise mean value.
        """
        if self.enabled:
            return self.caq_e_robust.summary()['value']
        else:
            return self.caq_e_mean.summary()['mean']

    def get_active_energy(self) -> float:
        """
        Get active energy value for guardrail decisions.

        Returns robust value if enabled, otherwise mean value.
        """
        if self.enabled:
            return self.energy_robust.summary()['value']
        else:
            return self.energy_mean.summary()['mean']

    def get_active_variance_caq_e(self) -> float:
        """Get variance for CAQ-E (from active aggregator)."""
        if self.enabled:
            return self.caq_e_robust.summary()['variance']
        else:
            return self.caq_e_mean.summary()['variance']

    def get_active_variance_energy(self) -> float:
        """Get variance for energy (from active aggregator)."""
        if self.enabled:
            return self.energy_robust.summary()['variance']
        else:
            return self.energy_mean.summary()['variance']

    def reset(self):
        """Reset all aggregators to initial state."""
        self.caq_e_mean.reset()
        self.energy_mean.reset()

        if self.enabled:
            self.caq_e_robust.reset()
            self.energy_robust.reset()

        self.update_count = 0

    def to_baseline_stats(self, config_name: str = "baseline") -> RobustBaselineStats:
        """
        Convert current state to RobustBaselineStats for persistence.

        Args:
            config_name: Configuration identifier

        Returns:
            RobustBaselineStats object
        """
        stats = self.get_current_stats()

        return RobustBaselineStats(
            mean_caq_e=stats['mean_caq_e'],
            std_caq_e=stats['std_caq_e'],
            mean_energy=stats['mean_energy'],
            std_energy=stats['std_energy'],
            config_name=config_name,
            num_samples=stats['n_samples'],
            robust_caq_e=stats['robust_caq_e'],
            robust_energy=stats['robust_energy'],
            robust_std_caq_e=stats['robust_std_caq_e'],
            robust_std_energy=stats['robust_std_energy'],
            robust_mode=stats['robust_mode']
        )


def compute_robust_baseline_from_data(
    caq_e_values: list,
    energy_values: list,
    robust_mode: str = 'auto',
    config_name: str = "baseline",
    window: int = None,
    seed: int = None
) -> RobustBaselineStats:
    """
    Compute robust baseline statistics from historical data.

    Convenience function for recomputing baselines from archived results.

    Args:
        caq_e_values: List of CAQ-E measurements
        energy_values: List of energy measurements
        robust_mode: Robust estimator mode
        config_name: Configuration identifier
        window: Rolling window size (None = full history)
        seed: Random seed for deterministic behavior

    Returns:
        RobustBaselineStats with both mean and robust statistics
    """
    adapter = RobustGuardrailAdapter(
        robust_mode=robust_mode,
        window=window,
        seed=seed
    )

    for caq_e, energy in zip(caq_e_values, energy_values):
        adapter.update(caq_e, energy)

    return adapter.to_baseline_stats(config_name=config_name)


if __name__ == '__main__':
    # Demonstration
    print("Robust Guardrail Adapter Demonstration")
    print("=" * 60)

    # Simulate contaminated CAQ-E data (like mixed gradients)
    np.random.seed(42)
    clean_caq = np.random.lognormal(mean=1.2, sigma=0.3, size=95)
    outlier_caq = np.random.lognormal(mean=3.0, sigma=0.5, size=5)
    caq_e_data = np.concatenate([clean_caq, outlier_caq])
    np.random.shuffle(caq_e_data)

    # Simulate energy data (correlated with CAQ-E)
    energy_data = caq_e_data * 3.5 + np.random.normal(0, 0.5, size=len(caq_e_data))

    # Test mean-only mode (legacy)
    print("\n1. Mean-Only Mode (Legacy Behavior)")
    print("-" * 60)
    adapter_mean = RobustGuardrailAdapter(robust_mode=None, window=20)

    for caq_e, energy in zip(caq_e_data, energy_data):
        adapter_mean.update(caq_e, energy)

    stats_mean = adapter_mean.get_current_stats()
    print(f"CAQ-E: {stats_mean['mean_caq_e']:.3f} ± {stats_mean['std_caq_e']:.3f}")
    print(f"Energy: {stats_mean['mean_energy']:.3f} ± {stats_mean['std_energy']:.3f}")
    print(f"Robust mode: {stats_mean['robust_mode']}")

    # Test robust mode (auto)
    print("\n2. Robust Mode (auto)")
    print("-" * 60)
    adapter_robust = RobustGuardrailAdapter(robust_mode='auto', window=20)

    for caq_e, energy in zip(caq_e_data, energy_data):
        adapter_robust.update(caq_e, energy)

    stats_robust = adapter_robust.get_current_stats()
    print(f"Mean CAQ-E: {stats_robust['mean_caq_e']:.3f} ± {stats_robust['std_caq_e']:.3f}")
    print(f"Robust CAQ-E: {stats_robust['robust_caq_e']:.3f} ± {stats_robust['robust_std_caq_e']:.3f}")
    print(f"Selected estimator: {stats_robust['selected_estimator_caq_e']}")

    print(f"\nMean Energy: {stats_robust['mean_energy']:.3f} ± {stats_robust['std_energy']:.3f}")
    print(f"Robust Energy: {stats_robust['robust_energy']:.3f} ± {stats_robust['robust_std_energy']:.3f}")

    # Show variance reduction
    var_reduction_caq = 1.0 - (stats_robust['robust_std_caq_e']**2 / stats_robust['std_caq_e']**2)
    print(f"\nVariance reduction (CAQ-E): {var_reduction_caq*100:.1f}%")

    print("\n" + "=" * 60)
    print("Robust adapter successfully integrated!")

#!/usr/bin/env python3
"""
Adaptive Confidence-Weighted Guardrail Decision Engine

Implements adaptive guardrail logic that uses per-layer uncertainty (σ) and
confidence intervals to weight and fuse per-layer CAQ-E signals into system
decisions. Reduces false positives while preserving sensitivity.

Phase D.4 - Adaptive Confidence-Weighted Guardrails
Builds on Phase D.2 (per-layer thresholds) and D.3 (uncertainty quantification).

Algorithm:
1. Compute standardized deviation: z_i = (x_i - θ_i) / (σ_i + ε)
2. Compute confidence weight: w_i = exp(-λ * σ_i / σ_ref) or inverse variance
3. Fuse decisions: S = Σ w_i * ReLU(z_i)
4. Trigger if S >= S_threshold or any z_i > max_layer_z

Author: Phase D.4 Implementation
License: MIT
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np


@dataclass
class LayerSignal:
    """Per-layer signal for adaptive guardrail."""
    layer_name: str
    observed_value: float
    threshold: float
    uncertainty: float
    confidence_lower: float
    confidence_upper: float
    standardized_deviation: float  # z_i
    weight: float  # w_i
    contribution: float  # w_i * ReLU(z_i)


@dataclass
class GuardrailDecision:
    """Adaptive guardrail decision result."""
    timestamp: float
    triggered: bool
    aggregate_score: float  # S
    threshold: float  # S_threshold
    reason: str
    top_contributors: List[Dict[str, Any]]
    all_layers: List[LayerSignal]
    decision_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'triggered': self.triggered,
            'aggregate_score': self.aggregate_score,
            'threshold': self.threshold,
            'reason': self.reason,
            'top_contributors': self.top_contributors,
            'all_layers': [asdict(layer) for layer in self.all_layers],
            'decision_latency_ms': self.decision_latency_ms
        }


class AdaptiveGuardrail:
    """
    Adaptive confidence-weighted guardrail decision engine.

    Uses per-layer uncertainties from D.3 to weight layer signals,
    reducing false positives while maintaining sensitivity.
    """

    def __init__(
        self,
        policy_config: Dict[str, Any],
        per_layer_thresholds: Dict[str, Any],
        uncertainty_estimates: Dict[str, Any],
        weight_mapping: str = "exponential",
        smoothing_tau: int = 1,
        seed: int = 42
    ):
        """
        Initialize adaptive guardrail.

        Args:
            policy_config: Adaptive policy configuration
            per_layer_thresholds: D.2 per-layer thresholds
            uncertainty_estimates: D.3 uncertainty estimates
            weight_mapping: Weight computation method ("exponential" or "inverse_variance")
            smoothing_tau: Temporal smoothing window size
            seed: Random seed for determinism
        """
        self.policy = policy_config
        self.layer_thresholds = per_layer_thresholds
        self.uncertainties = uncertainty_estimates
        self.weight_mapping = weight_mapping
        self.seed = seed

        # Extract policy parameters (with fallback to constructor arg)
        self.lambda_param = policy_config.get("lambda", 1.0)
        self.S_threshold = policy_config.get("S_threshold", 1.0)
        self.max_layer_z = policy_config.get("max_layer_z", 4.0)
        self.smoothing_tau = policy_config.get("smoothing_tau", smoothing_tau)

        # Compute reference σ for exponential mapping
        all_sigmas = [unc.get("std", 1.0) for unc in uncertainty_estimates.values()]
        self.sigma_ref = np.median(all_sigmas) if all_sigmas else 1.0

        # Smoothing buffer
        self.score_history = deque(maxlen=self.smoothing_tau)

        # Decision log
        self.decision_log = []

    def compute_weight(self, sigma: float) -> float:
        """
        Compute confidence weight from uncertainty.

        Higher uncertainty → lower weight.

        Args:
            sigma: Uncertainty (standard deviation)

        Returns:
            Weight value (0 to 1)
        """
        eps = 1e-9

        if self.weight_mapping == "exponential":
            # w_i = exp(-λ * σ_i / σ_ref)
            weight = np.exp(-self.lambda_param * sigma / (self.sigma_ref + eps))
        elif self.weight_mapping == "inverse_variance":
            # w_i = 1 / (σ_i^2 + α)
            alpha = 0.01
            weight = 1.0 / (sigma ** 2 + alpha)
        else:
            raise ValueError(f"Unknown weight mapping: {self.weight_mapping}")

        return float(weight)

    def compute_layer_signal(
        self,
        layer_name: str,
        observed_value: float
    ) -> LayerSignal:
        """
        Compute per-layer signal with weights.

        Args:
            layer_name: Name of the layer
            observed_value: Observed CAQ-E value

        Returns:
            LayerSignal object
        """
        # Get threshold from D.2
        layer_config = self.layer_thresholds.get(layer_name, {})
        threshold = layer_config.get("allowed_drift_pct", 0.15)

        # Get uncertainty from D.3
        layer_unc = self.uncertainties.get(layer_name, {})
        sigma = layer_unc.get("std", 1.0)
        ci_lower = layer_unc.get("ci_lower", observed_value - 1.96 * sigma)
        ci_upper = layer_unc.get("ci_upper", observed_value + 1.96 * sigma)

        # Compute standardized deviation
        eps = 1e-9
        z_i = (observed_value - threshold) / (sigma + eps)

        # Compute weight
        w_i = self.compute_weight(sigma)

        # Compute contribution (w_i * ReLU(z_i))
        contribution = w_i * max(0, z_i)

        return LayerSignal(
            layer_name=layer_name,
            observed_value=observed_value,
            threshold=threshold,
            uncertainty=sigma,
            confidence_lower=ci_lower,
            confidence_upper=ci_upper,
            standardized_deviation=z_i,
            weight=w_i,
            contribution=contribution
        )

    def update(
        self,
        per_layer_stats: Dict[str, float]
    ) -> GuardrailDecision:
        """
        Update adaptive guardrail with new per-layer observations.

        Args:
            per_layer_stats: Dictionary mapping layer names to observed values

        Returns:
            GuardrailDecision object
        """
        start_time = time.perf_counter()

        # Compute layer signals
        layer_signals = []
        for layer_name, value in per_layer_stats.items():
            signal = self.compute_layer_signal(layer_name, value)
            layer_signals.append(signal)

        # Normalize weights
        total_weight = sum(s.weight for s in layer_signals)
        if total_weight > 0:
            for signal in layer_signals:
                signal.weight /= total_weight
                signal.contribution = signal.weight * max(0, signal.standardized_deviation)

        # Compute aggregate score
        S = sum(s.contribution for s in layer_signals)

        # Apply temporal smoothing
        self.score_history.append(S)
        smoothed_S = np.mean(list(self.score_history))

        # Check trigger conditions
        triggered = False
        reason = "normal"

        # Condition 1: Smoothed score exceeds threshold
        if smoothed_S >= self.S_threshold:
            triggered = True
            reason = "adaptive_threshold_exceeded"

        # Condition 2: Any layer exceeds hard safety cap
        for signal in layer_signals:
            if signal.standardized_deviation > self.max_layer_z:
                triggered = True
                reason = f"safety_cap_exceeded_layer_{signal.layer_name}"
                break

        # Get top contributors
        sorted_signals = sorted(layer_signals, key=lambda s: s.contribution, reverse=True)
        top_contributors = [
            {
                'layer_name': s.layer_name,
                'contribution': float(s.contribution),
                'z_score': float(s.standardized_deviation),
                'weight': float(s.weight),
                'uncertainty': float(s.uncertainty)
            }
            for s in sorted_signals[:5]  # Top 5
        ]

        # Compute decision latency
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Create decision
        decision = GuardrailDecision(
            timestamp=time.time(),
            triggered=triggered,
            aggregate_score=float(smoothed_S),
            threshold=self.S_threshold,
            reason=reason,
            top_contributors=top_contributors,
            all_layers=layer_signals,
            decision_latency_ms=latency_ms
        )

        # Log decision
        self.decision_log.append(decision)

        return decision

    def decision_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent decisions.

        Returns:
            Summary dictionary
        """
        if not self.decision_log:
            return {
                'total_decisions': 0,
                'triggers': 0,
                'trigger_rate': 0.0,
                'avg_score': 0.0,
                'avg_latency_ms': 0.0
            }

        total = len(self.decision_log)
        triggers = sum(1 for d in self.decision_log if d.triggered)

        return {
            'total_decisions': total,
            'triggers': triggers,
            'trigger_rate': triggers / total if total > 0 else 0.0,
            'avg_score': np.mean([d.aggregate_score for d in self.decision_log]),
            'avg_latency_ms': np.mean([d.decision_latency_ms for d in self.decision_log]),
            'recent_decisions': [d.to_dict() for d in self.decision_log[-10:]]
        }

    def reset(self):
        """Reset decision state."""
        self.score_history.clear()
        self.decision_log.clear()


def load_adaptive_config(
    policy_path: Path,
    thresholds_path: Path,
    uncertainty_path: Path
) -> Tuple[Dict, Dict, Dict]:
    """
    Load configuration files for adaptive guardrail.

    Args:
        policy_path: Path to adaptive_policy.json
        thresholds_path: Path to per_layer_thresholds.json
        uncertainty_path: Path to propagated_confidence.json

    Returns:
        Tuple of (policy_config, layer_thresholds, uncertainty_estimates)
    """
    with open(policy_path, 'r') as f:
        policy = json.load(f)

    with open(thresholds_path, 'r') as f:
        thresholds_data = json.load(f)
        thresholds = thresholds_data.get("per_layer_thresholds", {})

    with open(uncertainty_path, 'r') as f:
        uncertainties = json.load(f)

    return policy, thresholds, uncertainties


if __name__ == "__main__":
    # Example usage
    print("Adaptive Guardrail Decision Engine - Phase D.4")
    print("This module provides adaptive confidence-weighted guardrail decisions.")
    print("Import and use AdaptiveGuardrail class for integration.")

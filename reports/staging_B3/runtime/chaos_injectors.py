#!/usr/bin/env python3
"""
Chaos Injectors for Phase B.2 Guardrail Stress Testing

Provides controlled anomaly injection for testing guardrail resilience.
All injectors modify in-process data only; no persistent system changes.

Safety:
  - No network calls
  - No permanent system modifications
  - All actions reversible
  - Offline-only operation
"""

import time
import random
import math
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json


logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be injected."""
    ENERGY_SPIKE = "energy_spike"
    ZERO_CAQ = "zero_caq"
    GRADUAL_DRIFT = "gradual_drift"
    EXCESSIVE_VARIANCE = "excessive_variance"
    NON_FINITE = "non_finite"
    COMBINED = "combined"


@dataclass
class AnomalyEvent:
    """Record of an injected anomaly."""
    timestamp: float
    anomaly_type: AnomalyType
    duration_ms: float
    magnitude: float
    target_metric: str
    correlation_id: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class InjectionConfig:
    """Configuration for chaos injection."""
    enabled: bool = False
    scenario: str = "baseline"
    seed: int = 42
    dry_run: bool = False
    log_file: Optional[Path] = None

    # Scenario-specific parameters
    spike_frequency_hz: float = 0.5  # spikes per second
    spike_magnitude: float = 2.0      # multiplier
    drift_rate: float = 0.01          # per-sample drift rate
    variance_noise_std: float = 0.5   # noise standard deviation

    def __post_init__(self):
        if self.log_file:
            self.log_file = Path(self.log_file)


class ChaosInjector:
    """
    Base class for chaos injection.

    Provides controlled anomaly injection for testing guardrail robustness.
    All modifications are in-process only; no system-wide changes.
    """

    def __init__(self, config: InjectionConfig):
        self.config = config
        self.random = random.Random(config.seed)
        self.events: List[AnomalyEvent] = []
        self.start_time = time.time()
        self.active_anomalies: Dict[str, AnomalyEvent] = {}

        # Initialize logging
        if config.log_file:
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(config.log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s'
            ))
            logger.addHandler(fh)
            logger.setLevel(logging.DEBUG)

    def inject_energy_spike(
        self,
        energy_value: float,
        magnitude: float = 2.0,
        correlation_id: str = ""
    ) -> Tuple[float, Optional[AnomalyEvent]]:
        """
        Inject a sudden energy spike.

        Args:
            energy_value: Original energy reading
            magnitude: Spike multiplier (2.0 = double energy)
            correlation_id: ID for tracking this event

        Returns:
            (modified_energy, event_record)
        """
        if not self.config.enabled or self.config.dry_run:
            return energy_value, None

        # Randomly decide if spike occurs based on frequency
        dt = time.time() - self.start_time
        if self.random.random() < self.config.spike_frequency_hz / 10:  # per update
            event = AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.ENERGY_SPIKE,
                duration_ms=random.uniform(500, 2000),
                magnitude=magnitude,
                target_metric="energy_joules",
                correlation_id=correlation_id or f"spike_{len(self.events)}"
            )
            self.events.append(event)

            modified = energy_value * magnitude
            logger.warning(
                f"[CHAOS] Energy spike injected: {energy_value:.4f} → {modified:.4f} "
                f"(×{magnitude:.1f}) | correlation_id={event.correlation_id}"
            )
            return modified, event

        return energy_value, None

    def inject_zero_caq(
        self,
        caq_value: float,
        correlation_id: str = ""
    ) -> Tuple[float, Optional[AnomalyEvent]]:
        """
        Inject a zero CAQ-E value (simulates sensor glitch).

        Args:
            caq_value: Original CAQ-E value
            correlation_id: ID for tracking

        Returns:
            (modified_caq, event_record)
        """
        if not self.config.enabled or self.config.dry_run:
            return caq_value, None

        # 1% chance of zero injection
        if self.random.random() < 0.01:
            event = AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.ZERO_CAQ,
                duration_ms=0,  # instant
                magnitude=0.0,
                target_metric="caq_e",
                correlation_id=correlation_id or f"zero_{len(self.events)}"
            )
            self.events.append(event)

            logger.warning(
                f"[CHAOS] Zero CAQ-E injected: {caq_value:.4f} → 0.0 | "
                f"correlation_id={event.correlation_id}"
            )
            return 0.0, event

        return caq_value, None

    def inject_gradual_drift(
        self,
        caq_value: float,
        baseline: float,
        elapsed_samples: int,
        correlation_id: str = ""
    ) -> Tuple[float, Optional[AnomalyEvent]]:
        """
        Inject gradual drift away from baseline.

        Args:
            caq_value: Original CAQ-E value
            baseline: Baseline CAQ-E mean
            elapsed_samples: Number of samples since start
            correlation_id: ID for tracking

        Returns:
            (modified_caq, event_record)
        """
        if not self.config.enabled or self.config.dry_run:
            return caq_value, None

        # Apply gradual drift: increase by drift_rate per sample
        drift_multiplier = 1.0 + (self.config.drift_rate * elapsed_samples)
        modified = baseline * drift_multiplier

        if elapsed_samples % 10 == 0:  # Log every 10 samples
            event = AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.GRADUAL_DRIFT,
                duration_ms=0,
                magnitude=drift_multiplier,
                target_metric="caq_e",
                correlation_id=correlation_id or f"drift_{len(self.events)}",
                metadata={"elapsed_samples": elapsed_samples, "baseline": baseline}
            )
            self.events.append(event)

            logger.debug(
                f"[CHAOS] Gradual drift: {caq_value:.2f} → {modified:.2f} "
                f"(×{drift_multiplier:.3f}) after {elapsed_samples} samples"
            )
            return modified, event

        return modified, None

    def inject_excessive_variance(
        self,
        caq_value: float,
        correlation_id: str = ""
    ) -> Tuple[float, Optional[AnomalyEvent]]:
        """
        Inject random noise to create excessive variance.

        Args:
            caq_value: Original CAQ-E value
            correlation_id: ID for tracking

        Returns:
            (modified_caq, event_record)
        """
        if not self.config.enabled or self.config.dry_run:
            return caq_value, None

        # Add random noise
        noise = self.random.gauss(0, self.config.variance_noise_std * caq_value)
        modified = max(0.01, caq_value + noise)  # Keep positive

        if abs(noise) > 0.3 * caq_value:  # Log significant noise
            event = AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.EXCESSIVE_VARIANCE,
                duration_ms=0,
                magnitude=abs(noise / caq_value),
                target_metric="caq_e",
                correlation_id=correlation_id or f"variance_{len(self.events)}",
                metadata={"noise": noise, "original": caq_value}
            )
            self.events.append(event)

            logger.debug(
                f"[CHAOS] Variance noise: {caq_value:.2f} → {modified:.2f} "
                f"(noise={noise:+.2f})"
            )
            return modified, event

        return modified, None

    def inject_non_finite(
        self,
        value: float,
        metric_name: str,
        correlation_id: str = ""
    ) -> Tuple[float, Optional[AnomalyEvent]]:
        """
        Inject NaN or Inf values.

        Args:
            value: Original value
            metric_name: Name of metric
            correlation_id: ID for tracking

        Returns:
            (modified_value, event_record)
        """
        if not self.config.enabled or self.config.dry_run:
            return value, None

        # 0.5% chance of non-finite injection
        if self.random.random() < 0.005:
            non_finite = self.random.choice([float('nan'), float('inf'), float('-inf')])
            event = AnomalyEvent(
                timestamp=time.time(),
                anomaly_type=AnomalyType.NON_FINITE,
                duration_ms=0,
                magnitude=0.0,
                target_metric=metric_name,
                correlation_id=correlation_id or f"nonfinite_{len(self.events)}",
                metadata={"original": value, "injected": str(non_finite)}
            )
            self.events.append(event)

            logger.warning(
                f"[CHAOS] Non-finite injected: {metric_name}={value:.4f} → {non_finite} | "
                f"correlation_id={event.correlation_id}"
            )
            return non_finite, event

        return value, None

    def apply_scenario(
        self,
        caq_value: float,
        energy_value: float,
        baseline_caq: float,
        elapsed_samples: int,
        correlation_id: str = ""
    ) -> Tuple[float, float, List[AnomalyEvent]]:
        """
        Apply scenario-specific chaos injection.

        Args:
            caq_value: Original CAQ-E
            energy_value: Original energy
            baseline_caq: Baseline CAQ-E mean
            elapsed_samples: Samples since start
            correlation_id: Tracking ID

        Returns:
            (modified_caq, modified_energy, events)
        """
        events = []

        if not self.config.enabled:
            return caq_value, energy_value, events

        scenario = self.config.scenario.lower()

        if scenario == "a_energy_spikes":
            energy_value, event = self.inject_energy_spike(
                energy_value,
                magnitude=self.config.spike_magnitude,
                correlation_id=correlation_id
            )
            if event:
                events.append(event)

        elif scenario == "b_zero_caq":
            caq_value, event = self.inject_zero_caq(caq_value, correlation_id)
            if event:
                events.append(event)

        elif scenario == "c_gradual_drift":
            caq_value, event = self.inject_gradual_drift(
                caq_value, baseline_caq, elapsed_samples, correlation_id
            )
            if event:
                events.append(event)

        elif scenario == "d_parallel_anomalies":
            # Combine energy spikes + zero CAQ
            energy_value, e1 = self.inject_energy_spike(energy_value, 2.5, correlation_id)
            caq_value, e2 = self.inject_zero_caq(caq_value, correlation_id)
            if e1:
                events.append(e1)
            if e2:
                events.append(e2)

        elif scenario == "e_excessive_variance":
            caq_value, event = self.inject_excessive_variance(caq_value, correlation_id)
            if event:
                events.append(event)

        elif scenario == "f_concurrency":
            # Similar to parallel but with worker-specific correlation
            energy_value, e1 = self.inject_energy_spike(energy_value, 1.8, correlation_id)
            if e1:
                events.append(e1)

        return caq_value, energy_value, events

    def get_timeline(self) -> List[Dict]:
        """Get timeline of all injected events."""
        return [
            {
                "timestamp": event.timestamp,
                "elapsed_ms": (event.timestamp - self.start_time) * 1000,
                "type": event.anomaly_type.value,
                "target": event.target_metric,
                "magnitude": event.magnitude,
                "duration_ms": event.duration_ms,
                "correlation_id": event.correlation_id,
                "metadata": event.metadata
            }
            for event in self.events
        ]

    def save_timeline(self, output_path: Path):
        """Save timeline to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        timeline = {
            "scenario": self.config.scenario,
            "seed": self.config.seed,
            "start_time": self.start_time,
            "total_events": len(self.events),
            "events": self.get_timeline()
        }

        with open(output_path, 'w') as f:
            json.dump(timeline, f, indent=2)

        logger.info(f"Timeline saved: {output_path} ({len(self.events)} events)")

    def reset(self):
        """Reset injector state."""
        self.events.clear()
        self.active_anomalies.clear()
        self.start_time = time.time()
        logger.info("Chaos injector reset")


class ScenarioFactory:
    """Factory for creating scenario-specific injection configurations."""

    @staticmethod
    def create_scenario(
        scenario_name: str,
        seed: int = 42,
        dry_run: bool = False,
        log_file: Optional[Path] = None
    ) -> InjectionConfig:
        """
        Create injection config for a named scenario.

        Scenarios:
          A: High-frequency jitter (energy spikes)
          B: Sudden drop to zeros
          C: Gradual drift
          D: Parallel anomalies
          E: Excessive variance
          F: Concurrency
          G: State persistence (no injection, tests crash recovery)
        """
        base_config = InjectionConfig(
            enabled=True,
            scenario=scenario_name,
            seed=seed,
            dry_run=dry_run,
            log_file=log_file
        )

        if scenario_name == "a_energy_spikes":
            base_config.spike_frequency_hz = 1.0  # 1 spike per second
            base_config.spike_magnitude = 2.5

        elif scenario_name == "b_zero_caq":
            # Use default config; inject_zero_caq has built-in probability
            pass

        elif scenario_name == "c_gradual_drift":
            base_config.drift_rate = 0.02  # 2% per sample

        elif scenario_name == "d_parallel_anomalies":
            base_config.spike_frequency_hz = 0.8
            base_config.spike_magnitude = 2.5

        elif scenario_name == "e_excessive_variance":
            base_config.variance_noise_std = 0.6  # 60% noise

        elif scenario_name == "f_concurrency":
            base_config.spike_frequency_hz = 0.5
            base_config.spike_magnitude = 1.8

        elif scenario_name == "g_state_persistence":
            base_config.enabled = False  # No injection; tests crash recovery

        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        return base_config


# Convenience functions for quick testing

def create_injector(scenario: str, seed: int = 42, **kwargs) -> ChaosInjector:
    """Create a chaos injector for the given scenario."""
    config = ScenarioFactory.create_scenario(scenario, seed, **kwargs)
    return ChaosInjector(config)


def inject_anomaly(
    injector: ChaosInjector,
    anomaly_type: AnomalyType,
    value: float,
    **kwargs
) -> Tuple[float, Optional[AnomalyEvent]]:
    """
    Inject a specific anomaly type.

    Args:
        injector: ChaosInjector instance
        anomaly_type: Type of anomaly to inject
        value: Original value
        **kwargs: Additional arguments for injection method

    Returns:
        (modified_value, event)
    """
    if anomaly_type == AnomalyType.ENERGY_SPIKE:
        return injector.inject_energy_spike(value, **kwargs)
    elif anomaly_type == AnomalyType.ZERO_CAQ:
        return injector.inject_zero_caq(value, **kwargs)
    elif anomaly_type == AnomalyType.NON_FINITE:
        return injector.inject_non_finite(value, kwargs.get('metric_name', 'unknown'), **kwargs)
    elif anomaly_type == AnomalyType.EXCESSIVE_VARIANCE:
        return injector.inject_excessive_variance(value, **kwargs)
    else:
        raise ValueError(f"Unsupported anomaly type: {anomaly_type}")


if __name__ == '__main__':
    # Demo usage
    logging.basicConfig(level=logging.DEBUG)

    print("=== Chaos Injector Demo ===\n")

    # Create injector for scenario A (energy spikes)
    config = ScenarioFactory.create_scenario("a_energy_spikes", seed=42)
    injector = ChaosInjector(config)

    print(f"Scenario: {config.scenario}")
    print(f"Spike frequency: {config.spike_frequency_hz} Hz")
    print(f"Spike magnitude: {config.spike_magnitude}x\n")

    # Simulate 20 updates
    for i in range(20):
        caq_e = 90.0
        energy = 0.05

        modified_caq, modified_energy, events = injector.apply_scenario(
            caq_e, energy, baseline_caq=90.0, elapsed_samples=i,
            correlation_id=f"test_{i}"
        )

        if events:
            print(f"[{i:02d}] Anomaly detected: {events[0].anomaly_type.value}")

        time.sleep(0.1)

    print(f"\nTotal events: {len(injector.events)}")
    print(f"Timeline entries: {len(injector.get_timeline())}")

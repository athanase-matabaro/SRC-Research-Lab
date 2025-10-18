#!/usr/bin/env python3
"""
Stress Runner for Phase B.2 Guardrail Testing

Executes stress scenarios to verify guardrail robustness, measure detection
latency, rollback time, and recovery performance.

Usage:
    python3 stress_runner.py --scenario a_energy_spikes --repeats 3
    python3 stress_runner.py --all-scenarios --output reports/b2_stress
    python3 stress_runner.py --dry-run --scenario c_gradual_drift
"""

import sys
import time
import argparse
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import statistics

# Add src-research-lab to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.guardrails import GuardrailManager, BaselineStats
from runtime.chaos_injectors import ChaosInjector, ScenarioFactory, AnomalyType


logger = logging.getLogger(__name__)


@dataclass
class StressMetrics:
    """Metrics collected during stress testing."""
    scenario: str
    run_id: int
    seed: int

    # Timing metrics (milliseconds)
    detection_latency_ms: List[float] = field(default_factory=list)
    rollback_time_ms: List[float] = field(default_factory=list)
    recovery_time_ms: List[float] = field(default_factory=list)

    # Accuracy metrics
    total_anomalies_injected: int = 0
    anomalies_detected: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Guardrail events
    total_updates: int = 0
    rollback_triggers: int = 0
    stability_checks: int = 0
    variance_violations: int = 0
    drift_violations: int = 0

    # State tracking
    final_stable: bool = False
    recovery_caq_e: Optional[float] = None
    max_drift_observed: float = 0.0
    max_variance_observed: float = 0.0

    def false_positive_rate(self) -> float:
        """Calculate false positive rate."""
        clean_samples = self.total_updates - self.total_anomalies_injected
        if clean_samples == 0:
            return 0.0
        return self.false_positives / clean_samples

    def false_negative_rate(self) -> float:
        """Calculate false negative rate."""
        if self.total_anomalies_injected == 0:
            return 0.0
        return self.false_negatives / self.total_anomalies_injected

    def median_detection_latency(self) -> float:
        """Median detection latency."""
        return statistics.median(self.detection_latency_ms) if self.detection_latency_ms else 0.0

    def median_rollback_time(self) -> float:
        """Median rollback time."""
        return statistics.median(self.rollback_time_ms) if self.rollback_time_ms else 0.0

    def median_recovery_time(self) -> float:
        """Median recovery time."""
        return statistics.median(self.recovery_time_ms) if self.recovery_time_ms else 0.0


@dataclass
class TimelineEvent:
    """Event in the stress test timeline."""
    timestamp: float
    elapsed_ms: float
    event_type: str
    details: Dict
    correlation_id: str = ""


class StressTestHarness:
    """
    Harness for running guardrail stress tests.

    Executes scenarios, injects anomalies, monitors guardrail responses,
    and collects performance metrics.
    """

    def __init__(
        self,
        baseline_stats: BaselineStats,
        output_dir: Path,
        dry_run: bool = False
    ):
        self.baseline_stats = baseline_stats
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run

        # Create output directories
        (self.output_dir / "timelines").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Initialize logging
        log_file = self.output_dir / "logs" / "stress_runner.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s'
        ))
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)

        logger.info("="*80)
        logger.info("STRESS TEST HARNESS INITIALIZED")
        logger.info("="*80)
        logger.info(f"Baseline CAQ-E: {baseline_stats.mean_caq_e:.2f} ± {baseline_stats.std_caq_e:.2f}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Dry run: {dry_run}")

    def run_scenario(
        self,
        scenario_name: str,
        run_id: int = 1,
        seed: Optional[int] = None,
        num_samples: int = 100
    ) -> StressMetrics:
        """
        Run a single stress scenario.

        Args:
            scenario_name: Name of scenario (a_energy_spikes, b_zero_caq, etc.)
            run_id: Run identifier
            seed: Random seed
            num_samples: Number of samples to collect

        Returns:
            StressMetrics with collected data
        """
        if seed is None:
            seed = 42 + run_id

        logger.info(f"\n{'='*80}")
        logger.info(f"SCENARIO: {scenario_name} | Run {run_id} | Seed {seed}")
        logger.info(f"{'='*80}")

        # Create chaos injector
        chaos_log = self.output_dir / "logs" / f"chaos_{scenario_name}_run{run_id}.log"
        config = ScenarioFactory.create_scenario(
            scenario_name,
            seed=seed,
            dry_run=self.dry_run,
            log_file=chaos_log
        )
        injector = ChaosInjector(config)

        # Create guardrail manager
        state_file = self.output_dir / f"guardrail_state_{scenario_name}_run{run_id}.json"
        guardrail = GuardrailManager(
            baseline_stats=self.baseline_stats,
            window=20,
            drift_threshold=0.15,
            variance_threshold=0.75,
            state_file=state_file
        )

        # Metrics collection
        metrics = StressMetrics(
            scenario=scenario_name,
            run_id=run_id,
            seed=seed
        )
        timeline: List[TimelineEvent] = []
        start_time = time.time()

        # Track anomaly injection timing
        anomaly_onset_times: Dict[str, float] = {}
        last_stable_time = start_time

        logger.info(f"Running {num_samples} samples...")

        for i in range(num_samples):
            sample_start = time.time()
            correlation_id = f"{scenario_name}_r{run_id}_s{i:03d}"

            # Generate baseline values (simulate benchmark)
            base_caq = self.baseline_stats.mean_caq_e
            base_energy = self.baseline_stats.mean_energy

            # Apply chaos injection
            modified_caq, modified_energy, chaos_events = injector.apply_scenario(
                caq_value=base_caq,
                energy_value=base_energy,
                baseline_caq=base_caq,
                elapsed_samples=i,
                correlation_id=correlation_id
            )

            # Track injected anomalies
            if chaos_events:
                metrics.total_anomalies_injected += len(chaos_events)
                for event in chaos_events:
                    anomaly_onset_times[event.correlation_id] = time.time()
                    timeline.append(TimelineEvent(
                        timestamp=time.time(),
                        elapsed_ms=(time.time() - start_time) * 1000,
                        event_type="anomaly_injected",
                        details={
                            "type": event.anomaly_type.value,
                            "target": event.target_metric,
                            "magnitude": event.magnitude
                        },
                        correlation_id=event.correlation_id
                    ))

            # Update guardrail
            try:
                update_result = guardrail.update_metrics(modified_caq, modified_energy)
                metrics.total_updates += 1

                timeline.append(TimelineEvent(
                    timestamp=time.time(),
                    elapsed_ms=(time.time() - start_time) * 1000,
                    event_type="guardrail_update",
                    details={
                        "caq_e": modified_caq,
                        "energy": modified_energy,
                        "variance": update_result.get("variance_percent", 0),
                        "drift": update_result.get("drift_percent", 0)
                    },
                    correlation_id=correlation_id
                ))

            except ValueError as e:
                # Guardrail rejected invalid input (expected for non-finite)
                logger.debug(f"Guardrail rejected input: {e}")
                continue

            # Check stability
            stability = guardrail.check_stability()
            metrics.stability_checks += 1

            if not stability["stable"]:
                # Guardrail triggered!
                trigger_time = time.time()
                metrics.anomalies_detected += 1

                # Calculate detection latency
                if chaos_events and chaos_events[0].correlation_id in anomaly_onset_times:
                    onset_time = anomaly_onset_times[chaos_events[0].correlation_id]
                    latency_ms = (trigger_time - onset_time) * 1000
                    metrics.detection_latency_ms.append(latency_ms)

                    logger.info(
                        f"[{i:03d}] Guardrail TRIGGERED | "
                        f"Latency: {latency_ms:.1f}ms | "
                        f"Violations: {', '.join(stability['violations'])}"
                    )
                elif not chaos_events:
                    # False positive (triggered without anomaly)
                    metrics.false_positives += 1
                    logger.warning(f"[{i:03d}] FALSE POSITIVE: {stability['violations']}")

                # Track violation types
                for violation in stability["violations"]:
                    if "Variance" in violation or "variance" in violation:
                        metrics.variance_violations += 1
                    if "Drift" in violation or "drift" in violation:
                        metrics.drift_violations += 1

                timeline.append(TimelineEvent(
                    timestamp=trigger_time,
                    elapsed_ms=(trigger_time - start_time) * 1000,
                    event_type="guardrail_triggered",
                    details={
                        "violations": stability["violations"],
                        "variance": stability.get("current_variance_percent", 0),
                        "drift": stability.get("current_drift_percent", 0)
                    },
                    correlation_id=correlation_id
                ))

                # Trigger rollback
                rollback_start = time.time()
                rollback_result = guardrail.trigger_rollback(
                    reason=f"{scenario_name}: {'; '.join(stability['violations'])}"
                )
                rollback_end = time.time()

                rollback_time_ms = (rollback_end - rollback_start) * 1000
                metrics.rollback_time_ms.append(rollback_time_ms)
                metrics.rollback_triggers += 1

                timeline.append(TimelineEvent(
                    timestamp=rollback_end,
                    elapsed_ms=(rollback_end - start_time) * 1000,
                    event_type="rollback_completed",
                    details={"rollback_time_ms": rollback_time_ms},
                    correlation_id=correlation_id
                ))

                logger.info(f"[{i:03d}] Rollback completed in {rollback_time_ms:.1f}ms")

                # Attempt recovery
                recovery_start = time.time()
                recovered = False
                recovery_samples = 0

                for j in range(50):  # Max 50 samples for recovery
                    # Inject clean data for recovery
                    clean_caq = self.baseline_stats.mean_caq_e
                    clean_energy = self.baseline_stats.mean_energy

                    try:
                        guardrail.update_metrics(clean_caq, clean_energy)
                        recovery_samples += 1

                        # Check if recovered
                        check = guardrail.check_stability()
                        current_drift = abs(check.get("current_drift_percent", 100))

                        if check["stable"] and current_drift < 5.0:
                            recovery_end = time.time()
                            recovery_time_ms = (recovery_end - recovery_start) * 1000
                            metrics.recovery_time_ms.append(recovery_time_ms)
                            recovered = True

                            logger.info(
                                f"[{i:03d}] RECOVERED in {recovery_time_ms:.1f}ms "
                                f"({recovery_samples} samples)"
                            )

                            timeline.append(TimelineEvent(
                                timestamp=recovery_end,
                                elapsed_ms=(recovery_end - start_time) * 1000,
                                event_type="recovery_completed",
                                details={
                                    "recovery_time_ms": recovery_time_ms,
                                    "recovery_samples": recovery_samples
                                },
                                correlation_id=correlation_id
                            ))
                            break

                    except ValueError:
                        continue

                if not recovered:
                    logger.warning(f"[{i:03d}] Failed to recover after {recovery_samples} samples")

            else:
                # Stable - check for false negatives
                if chaos_events:
                    # Anomaly was injected but not detected
                    metrics.false_negatives += 1
                    logger.warning(
                        f"[{i:03d}] FALSE NEGATIVE: Anomaly injected but not detected"
                    )

            # Track max values
            if "current_drift_percent" in stability:
                metrics.max_drift_observed = max(
                    metrics.max_drift_observed,
                    abs(stability["current_drift_percent"])
                )
            if "current_variance_percent" in stability:
                metrics.max_variance_observed = max(
                    metrics.max_variance_observed,
                    stability["current_variance_percent"]
                )

            # Small delay between samples
            time.sleep(0.01)

        # Final state
        final_check = guardrail.check_stability()
        metrics.final_stable = final_check["stable"]
        metrics.recovery_caq_e = guardrail.baseline.mean_caq_e

        # Save timeline
        timeline_file = self.output_dir / "timelines" / f"{scenario_name}_run{run_id}.json"
        self._save_timeline(timeline, timeline_file, metrics)

        # Save chaos events
        injector.save_timeline(
            self.output_dir / "timelines" / f"{scenario_name}_run{run_id}_chaos.json"
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"SCENARIO COMPLETE: {scenario_name} | Run {run_id}")
        logger.info(f"{'='*80}")
        logger.info(f"Total updates: {metrics.total_updates}")
        logger.info(f"Anomalies injected: {metrics.total_anomalies_injected}")
        logger.info(f"Anomalies detected: {metrics.anomalies_detected}")
        logger.info(f"False positives: {metrics.false_positives} ({metrics.false_positive_rate()*100:.2f}%)")
        logger.info(f"False negatives: {metrics.false_negatives} ({metrics.false_negative_rate()*100:.2f}%)")
        logger.info(f"Median detection latency: {metrics.median_detection_latency():.1f}ms")
        logger.info(f"Median rollback time: {metrics.median_rollback_time():.1f}ms")
        logger.info(f"Median recovery time: {metrics.median_recovery_time():.1f}ms")

        return metrics

    def _save_timeline(self, timeline: List[TimelineEvent], output_path: Path, metrics: StressMetrics):
        """Save timeline to JSON file."""
        data = {
            "scenario": metrics.scenario,
            "run_id": metrics.run_id,
            "seed": metrics.seed,
            "total_events": len(timeline),
            "summary": {
                "total_updates": metrics.total_updates,
                "anomalies_injected": metrics.total_anomalies_injected,
                "anomalies_detected": metrics.anomalies_detected,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives
            },
            "events": [
                {
                    "timestamp": evt.timestamp,
                    "elapsed_ms": evt.elapsed_ms,
                    "type": evt.event_type,
                    "correlation_id": evt.correlation_id,
                    "details": evt.details
                }
                for evt in timeline
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Timeline saved: {output_path}")

    def run_all_scenarios(self, repeats: int = 3) -> Dict[str, List[StressMetrics]]:
        """
        Run all scenarios A-G with multiple repeats.

        Args:
            repeats: Number of repeats per scenario

        Returns:
            Dict mapping scenario name to list of metrics
        """
        scenarios = [
            "a_energy_spikes",
            "b_zero_caq",
            "c_gradual_drift",
            "d_parallel_anomalies",
            "e_excessive_variance",
            "f_concurrency",
            # Note: g_state_persistence requires special handling (process kill)
        ]

        all_metrics = {}

        for scenario in scenarios:
            logger.info(f"\n\n{'#'*80}")
            logger.info(f"# STARTING SCENARIO: {scenario.upper()}")
            logger.info(f"# Repeats: {repeats}")
            logger.info(f"{'#'*80}\n")

            scenario_metrics = []
            for run in range(1, repeats + 1):
                metrics = self.run_scenario(
                    scenario_name=scenario,
                    run_id=run,
                    seed=42 + run,
                    num_samples=100
                )
                scenario_metrics.append(metrics)

            all_metrics[scenario] = scenario_metrics

        return all_metrics


def generate_summary_report(
    all_metrics: Dict[str, List[StressMetrics]],
    output_dir: Path
):
    """Generate summary CSV and JSON reports."""
    output_dir = Path(output_dir)

    # Generate latency CSV
    latency_csv = output_dir / "latency.csv"
    with open(latency_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "run_id", "seed",
            "median_detection_ms", "median_rollback_ms", "median_recovery_ms",
            "anomalies_injected", "anomalies_detected",
            "false_positive_rate", "false_negative_rate"
        ])

        for scenario, metrics_list in all_metrics.items():
            for metrics in metrics_list:
                writer.writerow([
                    metrics.scenario,
                    metrics.run_id,
                    metrics.seed,
                    metrics.median_detection_latency(),
                    metrics.median_rollback_time(),
                    metrics.median_recovery_time(),
                    metrics.total_anomalies_injected,
                    metrics.anomalies_detected,
                    metrics.false_positive_rate(),
                    metrics.false_negative_rate()
                ])

    logger.info(f"Latency CSV saved: {latency_csv}")

    # Generate recovery JSON
    recovery_data = {}
    for scenario, metrics_list in all_metrics.items():
        recovery_data[scenario] = [
            {
                "run_id": m.run_id,
                "final_stable": m.final_stable,
                "recovery_caq_e": m.recovery_caq_e,
                "max_drift": m.max_drift_observed,
                "max_variance": m.max_variance_observed,
                "recovery_times_ms": m.recovery_time_ms
            }
            for m in metrics_list
        ]

    recovery_json = output_dir / "caq_e_recovery.json"
    with open(recovery_json, 'w') as f:
        json.dump(recovery_data, f, indent=2)

    logger.info(f"Recovery JSON saved: {recovery_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase B.2 Guardrail Stress Testing"
    )
    parser.add_argument(
        "--scenario",
        choices=[
            "a_energy_spikes", "b_zero_caq", "c_gradual_drift",
            "d_parallel_anomalies", "e_excessive_variance", "f_concurrency",
            "g_state_persistence"
        ],
        help="Specific scenario to run"
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all scenarios"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per scenario (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/b2_stress"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without injecting anomalies"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per run (default: 100)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )

    # Create baseline stats
    baseline = BaselineStats(
        mean_caq_e=87.92,
        std_caq_e=42.62,
        mean_energy=0.041,
        std_energy=0.023,
        config_name="stress_test_baseline"
    )

    # Create harness
    harness = StressTestHarness(
        baseline_stats=baseline,
        output_dir=args.output,
        dry_run=args.dry_run
    )

    # Run scenarios
    if args.all_scenarios:
        all_metrics = harness.run_all_scenarios(repeats=args.repeats)
        generate_summary_report(all_metrics, args.output)
    elif args.scenario:
        metrics = harness.run_scenario(
            scenario_name=args.scenario,
            run_id=1,
            seed=42,
            num_samples=args.num_samples
        )
        print(f"\nResults: {args.output}")
    else:
        parser.error("Must specify --scenario or --all-scenarios")


if __name__ == '__main__':
    main()

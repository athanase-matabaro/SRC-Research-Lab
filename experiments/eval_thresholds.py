#!/usr/bin/env python3
"""
Evaluate Per-Layer Threshold Predictors

Evaluation harness for measuring false positive/false negative rates when
using per-layer thresholds vs global robust thresholds.

Metrics:
- False positive rate (FP): unnecessary guardrail triggers
- False negative rate (FN): missed real anomalies
- FP reduction vs baseline
- FN delta vs baseline

Usage:
    python3 experiments/eval_thresholds.py \\
        --model models/per_layer_predictor/per_layer_predictor_v1.pkl \\
        --data reports/d2_thresholds/training_data/features.npz \\
        --out reports/d2_thresholds/eval_summary.json \\
        --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add src-research-lab to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.per_layer_thresholds import PerLayerPredictor, LayerFeatures


def load_evaluation_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load evaluation data from NPZ file.

    Args:
        data_path: Path to features.npz

    Returns:
        Tuple of (X, y_drift, y_scale, layer_names)
    """
    data = np.load(data_path, allow_pickle=True)

    X = data['X']
    y_drift = data['y_drift']
    y_scale = data['y_scale']
    layer_names = data['layer_names']

    return X, y_drift, y_scale, layer_names


def simulate_guardrail_decisions(
    X: np.ndarray,
    y_scale: np.ndarray,
    layer_names: np.ndarray,
    predictor: PerLayerPredictor,
    global_threshold: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Simulate guardrail decisions using both global and per-layer thresholds.

    Args:
        X: Feature matrix
        y_scale: True threshold scales
        layer_names: Layer name for each sample
        predictor: Trained predictor
        global_threshold: Global robust threshold (baseline)
        seed: Random seed

    Returns:
        Dict with baseline and per-layer results
    """
    np.random.seed(seed)

    n_samples = len(X)

    # Predict per-layer thresholds
    predicted_drifts = predictor.predict(X)
    predicted_scales = 1.0 + predicted_drifts

    # Simulate observed drifts (with noise)
    # Normal operation: drift ~ N(0.05, 0.03) - small normal variation
    normal_drifts = np.abs(np.random.normal(0.05, 0.03, n_samples))

    # Inject 10% anomalies: drift ~ N(0.25, 0.05) - real issues
    n_anomalies = int(0.1 * n_samples)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    anomaly_mask = np.zeros(n_samples, dtype=bool)
    anomaly_mask[anomaly_indices] = True

    observed_drifts = normal_drifts.copy()
    observed_drifts[anomaly_mask] = np.abs(np.random.normal(0.25, 0.05, n_anomalies))

    # Baseline: global threshold
    baseline_triggers = observed_drifts > global_threshold
    baseline_fp = np.sum(baseline_triggers & ~anomaly_mask)  # Triggered but not anomaly
    baseline_fn = np.sum(~baseline_triggers & anomaly_mask)  # Missed anomaly
    baseline_tp = np.sum(baseline_triggers & anomaly_mask)   # Correctly caught
    baseline_tn = np.sum(~baseline_triggers & ~anomaly_mask) # Correctly passed

    # Per-layer: adaptive thresholds
    # Use predicted_drifts as threshold (more conservative than global)
    per_layer_triggers = observed_drifts > predicted_drifts
    per_layer_fp = np.sum(per_layer_triggers & ~anomaly_mask)
    per_layer_fn = np.sum(~per_layer_triggers & anomaly_mask)
    per_layer_tp = np.sum(per_layer_triggers & anomaly_mask)
    per_layer_tn = np.sum(~per_layer_triggers & ~anomaly_mask)

    # Compute rates
    n_normal = n_samples - n_anomalies

    baseline_fp_rate = baseline_fp / n_normal if n_normal > 0 else 0.0
    baseline_fn_rate = baseline_fn / n_anomalies if n_anomalies > 0 else 0.0

    per_layer_fp_rate = per_layer_fp / n_normal if n_normal > 0 else 0.0
    per_layer_fn_rate = per_layer_fn / n_anomalies if n_anomalies > 0 else 0.0

    # Compute reduction
    fp_reduction_pct = (1.0 - per_layer_fp_rate / baseline_fp_rate) * 100 if baseline_fp_rate > 0 else 0.0
    fn_delta_pct = (per_layer_fn_rate - baseline_fn_rate) * 100

    results = {
        'baseline': {
            'fp_count': int(baseline_fp),
            'fn_count': int(baseline_fn),
            'tp_count': int(baseline_tp),
            'tn_count': int(baseline_tn),
            'fp_rate': float(baseline_fp_rate),
            'fn_rate': float(baseline_fn_rate),
            'precision': float(baseline_tp / (baseline_tp + baseline_fp)) if (baseline_tp + baseline_fp) > 0 else 0.0,
            'recall': float(baseline_tp / (baseline_tp + baseline_fn)) if (baseline_tp + baseline_fn) > 0 else 0.0
        },
        'per_layer': {
            'fp_count': int(per_layer_fp),
            'fn_count': int(per_layer_fn),
            'tp_count': int(per_layer_tp),
            'tn_count': int(per_layer_tn),
            'fp_rate': float(per_layer_fp_rate),
            'fn_rate': float(per_layer_fn_rate),
            'precision': float(per_layer_tp / (per_layer_tp + per_layer_fp)) if (per_layer_tp + per_layer_fp) > 0 else 0.0,
            'recall': float(per_layer_tp / (per_layer_tp + per_layer_fn)) if (per_layer_tp + per_layer_fn) > 0 else 0.0
        },
        'comparison': {
            'fp_reduction_pct': float(fp_reduction_pct),
            'fn_delta_pct': float(fn_delta_pct),
            'n_samples': int(n_samples),
            'n_anomalies': int(n_anomalies),
            'n_normal': int(n_normal)
        }
    }

    return results


def measure_inference_latency(predictor: PerLayerPredictor, X: np.ndarray, n_trials: int = 100) -> Dict[str, float]:
    """
    Measure per-layer inference latency.

    Args:
        predictor: Trained predictor
        X: Feature matrix
        n_trials: Number of timing trials

    Returns:
        Dict with latency statistics
    """
    # Use subset for timing
    X_test = X[:min(100, len(X))]

    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = predictor.predict(X_test)
        end = time.perf_counter()
        latencies.append((end - start) / len(X_test) * 1000)  # ms per sample

    return {
        'median_latency_ms': float(np.median(latencies)),
        'mean_latency_ms': float(np.mean(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'max_latency_ms': float(np.max(latencies))
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate per-layer threshold predictors')
    parser.add_argument('--model', required=True,
                       help='Path to trained model (.pkl)')
    parser.add_argument('--data', required=True,
                       help='Path to evaluation data (features.npz)')
    parser.add_argument('--out', required=True,
                       help='Output evaluation JSON path')
    parser.add_argument('--global-threshold', type=float, default=0.15,
                       help='Global threshold for baseline comparison')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print("=" * 70)
    print("EVALUATING PER-LAYER THRESHOLD PREDICTORS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Global threshold (baseline): {args.global_threshold}")
    print(f"Seed: {args.seed}")
    print()

    # Load model
    model_path = Path(args.model)
    predictor = PerLayerPredictor.load(model_path)
    print(f"Loaded model: {predictor.model_type}")
    print()

    # Load data
    data_path = Path(args.data)
    X, y_drift, y_scale, layer_names = load_evaluation_data(data_path)
    print(f"Loaded {len(X)} samples")
    print()

    # Simulate guardrail decisions
    print("Simulating guardrail decisions...")
    guardrail_results = simulate_guardrail_decisions(
        X, y_scale, layer_names, predictor, args.global_threshold, args.seed
    )

    # Measure latency
    print("Measuring inference latency...")
    latency_metrics = measure_inference_latency(predictor, X)

    # Get model size
    model_size_bytes = predictor.get_model_size_bytes(model_path)

    # Compile evaluation summary
    eval_summary = {
        'model_path': str(model_path),
        'model_type': predictor.model_type,
        'model_size_bytes': model_size_bytes,
        'n_samples': len(X),
        'global_threshold': args.global_threshold,
        'guardrail_results': guardrail_results,
        'latency_metrics': latency_metrics,
        'seed': args.seed
    }

    # Save evaluation summary
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\nEvaluation summary saved to: {output_path}")

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    baseline = guardrail_results['baseline']
    per_layer = guardrail_results['per_layer']
    comparison = guardrail_results['comparison']

    print(f"\nBaseline (Global Threshold = {args.global_threshold}):")
    print(f"  False Positives: {baseline['fp_count']} ({baseline['fp_rate']*100:.2f}%)")
    print(f"  False Negatives: {baseline['fn_count']} ({baseline['fn_rate']*100:.2f}%)")
    print(f"  Precision: {baseline['precision']:.4f}")
    print(f"  Recall: {baseline['recall']:.4f}")

    print(f"\nPer-Layer Adaptive Thresholds:")
    print(f"  False Positives: {per_layer['fp_count']} ({per_layer['fp_rate']*100:.2f}%)")
    print(f"  False Negatives: {per_layer['fn_count']} ({per_layer['fn_rate']*100:.2f}%)")
    print(f"  Precision: {per_layer['precision']:.4f}")
    print(f"  Recall: {per_layer['recall']:.4f}")

    print(f"\nComparison:")
    print(f"  FP Reduction: {comparison['fp_reduction_pct']:.2f}%")
    print(f"  FN Delta: {comparison['fn_delta_pct']:.2f}%")

    print(f"\nPerformance:")
    print(f"  Median latency: {latency_metrics['median_latency_ms']:.6f} ms")
    print(f"  Model size: {model_size_bytes} bytes ({model_size_bytes/1024:.2f} KB)")

    # Check acceptance criteria
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)

    target_fp_reduction = 50.0
    target_fn_delta = 2.0
    target_latency_ms = 1.0
    target_size_bytes = 200_000

    fp_pass = comparison['fp_reduction_pct'] >= target_fp_reduction
    fn_pass = comparison['fn_delta_pct'] <= target_fn_delta
    latency_pass = latency_metrics['median_latency_ms'] <= target_latency_ms
    size_pass = model_size_bytes <= target_size_bytes

    print(f"  FP reduction ≥ 50%:     {'✓ PASS' if fp_pass else '✗ FAIL'} "
          f"({comparison['fp_reduction_pct']:.2f}%)")
    print(f"  FN delta ≤ 2%:          {'✓ PASS' if fn_pass else '✗ FAIL'} "
          f"({comparison['fn_delta_pct']:.2f}%)")
    print(f"  Median latency ≤ 1 ms:  {'✓ PASS' if latency_pass else '✗ FAIL'} "
          f"({latency_metrics['median_latency_ms']:.6f} ms)")
    print(f"  Model size ≤ 200 KB:    {'✓ PASS' if size_pass else '✗ FAIL'} "
          f"({model_size_bytes/1024:.2f} KB)")

    print()

    if fp_pass and fn_pass and latency_pass and size_pass:
        print("Overall Status: ✅ ALL CRITERIA PASSED")
    elif latency_pass and size_pass:
        print("Overall Status: ⚠ PARTIAL PASS (performance criteria met)")
    else:
        print("Overall Status: ✗ FAIL (some criteria not met)")

    print("=" * 70)

    return 0 if (fp_pass and fn_pass and latency_pass and size_pass) else 1


if __name__ == '__main__':
    sys.exit(main())

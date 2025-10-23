#!/usr/bin/env python3
"""
Train Per-Layer Threshold Predictors

Training harness with cross-validation and hyperparameter search for
per-layer CAQ-E anomaly detection thresholds.

Features:
- K-fold cross-validation (k=5)
- Hyperparameter grid search
- Model selection (ridge/tree/mlp)
- Deterministic training with seed control
- Model size and latency tracking

Usage:
    python3 experiments/train_per_layer_thresholds.py \\
        --data reports/d2_thresholds/training_data/features.npz \\
        --model ridge \\
        --cv 5 \\
        --seed 42 \\
        --out models/per_layer_predictor/per_layer_predictor_v1.pkl \\
        --report reports/d2_thresholds/train_summary.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# Add src-research-lab to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.per_layer_thresholds import PerLayerPredictor


def load_training_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load training data from NPZ file.

    Args:
        data_path: Path to features.npz

    Returns:
        Tuple of (X, y, feature_names)
    """
    data = np.load(data_path, allow_pickle=True)

    X = data['X']

    # Use y_scale (threshold_scale) instead of y_drift (allowed_drift_pct)
    # because y_drift hits 0.50 cap for all samples (zero variance)
    if 'y_scale' in data:
        y = data['y_scale']
    else:
        y = data['y_drift']

    feature_names = data['feature_names'].tolist()

    return X, y, feature_names


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    alpha: float,
    max_depth: int,
    n_folds: int,
    seed: int
) -> Dict[str, float]:
    """
    Perform k-fold cross-validation.

    Args:
        X: Feature matrix
        y: Target labels
        model_type: 'ridge', 'tree', or 'mlp'
        alpha: L2 regularization
        max_depth: Max depth for decision tree
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        Dict with CV metrics (rmse, r2, etc.)
    """
    np.random.seed(seed)

    # Shuffle indices
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    # Split into folds
    fold_size = n_samples // n_folds
    fold_scores = []

    for fold in range(n_folds):
        # Create train/val split
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        predictor = PerLayerPredictor(
            model_type=model_type,
            alpha=alpha,
            max_depth=max_depth,
            seed=seed
        )
        predictor.fit(X_train, y_train)

        # Evaluate
        y_pred = predictor.predict(X_val)

        # Compute metrics
        mse = np.mean((y_val - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # R² score
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Mean absolute error
        mae = np.mean(np.abs(y_val - y_pred))

        fold_scores.append({
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        })

    # Average across folds
    cv_metrics = {
        'cv_rmse': float(np.mean([s['rmse'] for s in fold_scores])),
        'cv_r2': float(np.mean([s['r2'] for s in fold_scores])),
        'cv_mae': float(np.mean([s['mae'] for s in fold_scores])),
        'cv_rmse_std': float(np.std([s['rmse'] for s in fold_scores])),
        'cv_r2_std': float(np.std([s['r2'] for s in fold_scores])),
        'n_folds': n_folds
    }

    return cv_metrics


def hyperparameter_search(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_folds: int,
    seed: int
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Grid search for best hyperparameters.

    Args:
        X: Feature matrix
        y: Target labels
        model_type: 'ridge', 'tree', or 'mlp'
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        Tuple of (best_params, cv_results)
    """
    print(f"\nHyperparameter search for {model_type}...")

    if model_type == 'ridge':
        alpha_grid = [1e-3, 1e-2, 1e-1, 1.0]
        results = []

        for alpha in alpha_grid:
            print(f"  Testing alpha={alpha:.4f}...")
            cv_metrics = cross_validate(X, y, model_type, alpha, 4, n_folds, seed)
            results.append({'alpha': alpha, **cv_metrics})

        # Select best based on CV R²
        best_result = max(results, key=lambda x: x['cv_r2'])
        best_params = {'alpha': best_result['alpha'], 'max_depth': 4}

    elif model_type == 'tree':
        depth_grid = [2, 3, 4, 5]
        results = []

        for max_depth in depth_grid:
            print(f"  Testing max_depth={max_depth}...")
            cv_metrics = cross_validate(X, y, model_type, 0.01, max_depth, n_folds, seed)
            results.append({'max_depth': max_depth, **cv_metrics})

        best_result = max(results, key=lambda x: x['cv_r2'])
        best_params = {'alpha': 0.01, 'max_depth': best_result['max_depth']}

    elif model_type == 'mlp':
        alpha_grid = [1e-3, 1e-2]
        results = []

        for alpha in alpha_grid:
            print(f"  Testing alpha={alpha:.4f}...")
            cv_metrics = cross_validate(X, y, model_type, alpha, 4, n_folds, seed)
            results.append({'alpha': alpha, **cv_metrics})

        best_result = max(results, key=lambda x: x['cv_r2'])
        best_params = {'alpha': best_result['alpha'], 'max_depth': 4}

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return best_params, results


def measure_inference_latency(predictor: PerLayerPredictor, X: np.ndarray, n_trials: int = 100) -> Dict[str, float]:
    """
    Measure per-layer inference latency.

    Args:
        predictor: Trained predictor
        X: Feature matrix (use subset for timing)
        n_trials: Number of timing trials

    Returns:
        Dict with latency statistics
    """
    # Use first 100 samples
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


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    best_params: Dict[str, Any],
    seed: int,
    output_path: Path
) -> PerLayerPredictor:
    """
    Train final model on full dataset with best hyperparameters.

    Args:
        X: Feature matrix
        y: Target labels
        model_type: Model type
        best_params: Best hyperparameters from search
        seed: Random seed
        output_path: Output model path

    Returns:
        Trained predictor
    """
    print("\nTraining final model on full dataset...")

    predictor = PerLayerPredictor(
        model_type=model_type,
        alpha=best_params['alpha'],
        max_depth=best_params['max_depth'],
        seed=seed
    )

    predictor.fit(X, y)

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(output_path)

    model_size = predictor.get_model_size_bytes(output_path)
    print(f"Model saved to: {output_path}")
    print(f"Model size: {model_size} bytes ({model_size/1024:.2f} KB)")

    return predictor


def main():
    parser = argparse.ArgumentParser(description='Train per-layer threshold predictors')
    parser.add_argument('--data', required=True,
                       help='Path to training data (features.npz)')
    parser.add_argument('--model', default='ridge',
                       choices=['ridge', 'tree', 'mlp'],
                       help='Model type')
    parser.add_argument('--cv', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--out', required=True,
                       help='Output model path (.pkl)')
    parser.add_argument('--report', required=True,
                       help='Output report JSON path')

    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING PER-LAYER THRESHOLD PREDICTORS")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Model type: {args.model}")
    print(f"CV folds: {args.cv}")
    print(f"Seed: {args.seed}")
    print()

    # Load data
    data_path = Path(args.data)
    X, y, feature_names = load_training_data(data_path)

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Target (threshold_scale): mean={y.mean():.4f}, std={y.std():.4f}")
    print()

    # Hyperparameter search
    best_params, cv_results = hyperparameter_search(X, y, args.model, args.cv, args.seed)

    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best CV R²: {max(r['cv_r2'] for r in cv_results):.4f}")
    print()

    # Train final model
    output_path = Path(args.out)
    predictor = train_final_model(X, y, args.model, best_params, args.seed, output_path)

    # Measure latency
    print("\nMeasuring inference latency...")
    latency_metrics = measure_inference_latency(predictor, X)
    print(f"Median latency: {latency_metrics['median_latency_ms']:.6f} ms per layer")

    # Get feature importances
    importances = predictor.get_feature_importances()

    # Create training report
    report = {
        'model_type': args.model,
        'best_params': best_params,
        'cv_results': cv_results,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': feature_names,
        'feature_importances': importances,
        'latency_metrics': latency_metrics,
        'model_size_bytes': predictor.get_model_size_bytes(output_path),
        'seed': args.seed,
        'output_path': str(output_path)
    }

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nTraining report saved to: {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Model type: {args.model}")
    print(f"Best params: {best_params}")
    print(f"CV R²: {max(r['cv_r2'] for r in cv_results):.4f}")
    print(f"CV RMSE: {min(r['cv_rmse'] for r in cv_results):.4f}")
    print(f"Model size: {report['model_size_bytes']} bytes")
    print(f"Median latency: {latency_metrics['median_latency_ms']:.6f} ms")
    print()

    print("Top 5 Feature Importances:")
    for name, imp in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        print(f"  {name:20s}: {imp:.4f}")
    print()

    # Check acceptance criteria
    target_latency_ms = 1.0
    target_size_bytes = 200_000

    latency_pass = latency_metrics['median_latency_ms'] <= target_latency_ms
    size_pass = report['model_size_bytes'] <= target_size_bytes

    print("Acceptance Criteria:")
    print(f"  Median latency ≤ 1.0 ms: {'✓ PASS' if latency_pass else '✗ FAIL'} "
          f"({latency_metrics['median_latency_ms']:.6f} ms)")
    print(f"  Model size ≤ 200 KB:     {'✓ PASS' if size_pass else '✗ FAIL'} "
          f"({report['model_size_bytes']/1024:.2f} KB)")
    print()

    if latency_pass and size_pass:
        print("Overall Status: ✅ PASS")
    else:
        print("Overall Status: ⚠ MARGINAL (some criteria not met)")

    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())

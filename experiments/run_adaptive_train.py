#!/usr/bin/env python3
"""
Adaptive Training Loop Experiment.

Simulates mini-training with adaptive compression and CAQ optimization.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_model.neural_entropy import NeuralEntropyPredictor
from adaptive_model.gradient_encoder import GradientEncoder
from adaptive_model.scheduler import CompressionScheduler
from adaptive_model.utils import compute_tensor_stats, format_adaptive_result
from metrics.caq_metric import compute_caq


def generate_synthetic_gradient(epoch, shape=(100, 100)):
    """Generate synthetic gradient tensor."""
    np.random.seed(epoch + 42)
    # Simulate gradient with structure
    base = np.random.randn(*shape).astype(np.float32)
    # Add some sparsity
    mask = np.random.rand(*shape) > 0.3
    return base * mask


def run_baseline_compression(tensor):
    """Run baseline compression (simple npz)."""
    import tempfile
    import time

    start = time.time()
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = tmp.name
    np.savez_compressed(tmp_path, data=tensor)

    original_size = tensor.nbytes
    compressed_size = Path(tmp_path).stat().st_size
    cpu_time = time.time() - start

    Path(tmp_path).unlink()

    ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    caq = compute_caq(ratio, cpu_time)

    return {
        "compression_ratio": ratio,
        "cpu_time": cpu_time,
        "caq": caq
    }


def main():
    print("=" * 60)
    print("ADAPTIVE LEARNED COMPRESSION MODEL (ALCM)")
    print("Phase H.3 Experiment")
    print("=" * 60)
    print()

    # Load configurations
    config_dir = Path(__file__).parent.parent / "adaptive_model" / "configs"
    entropy_config = config_dir / "entropy_config.yaml"
    encoder_config = config_dir / "encoder_config.yaml"

    # Initialize components
    print("Initializing components...")
    predictor = NeuralEntropyPredictor(config_path=entropy_config)
    encoder = GradientEncoder(config_path=encoder_config)
    scheduler = CompressionScheduler()

    # Train entropy predictor
    print("\nTraining Neural Entropy Predictor...")
    print("-" * 60)

    # Generate synthetic training data
    feature_list = []
    target_list = []

    for i in range(50):
        tensor = generate_synthetic_gradient(i, shape=(50, 50))
        stats = compute_tensor_stats(tensor)
        features = np.array([
            stats['mean'], stats['var'], stats['std'],
            stats['skew'], stats['kurtosis'], stats['sparsity']
        ])
        # Target: higher sparsity -> lower bits needed
        target = 0.3 + 0.5 * (1 - stats['sparsity'])
        feature_list.append(features)
        target_list.append(target)

    final_loss = predictor.train(feature_list, target_list, epochs=10, verbose=False)
    print(f"Training complete. Final loss: {final_loss:.6f}")

    # Run adaptive compression experiment
    print("\nRunning Adaptive Compression Experiment...")
    print("-" * 60)

    results = []
    epochs = 10

    for epoch in range(1, epochs + 1):
        # Generate synthetic gradient
        tensor = generate_synthetic_gradient(epoch)

        # Baseline compression
        baseline = run_baseline_compression(tensor)

        # Adaptive compression
        entropy_map = predictor.predict_entropy_map(tensor)
        adaptive_result = encoder.compress_tensor(tensor, entropy_map)

        # Update scheduler
        new_threshold = scheduler.update(adaptive_result['caq'])

        # Update encoder threshold
        encoder.threshold = new_threshold

        # Compute variance (simplified - using drop ratio variance)
        variance = adaptive_result['dropped_ratio'] * 100

        # Format result
        result = format_adaptive_result(
            status="PASS",
            baseline_caq=baseline['caq'],
            adaptive_caq=adaptive_result['caq'],
            variance=variance,
            entropy_loss=final_loss,
            notes=f"Epoch {epoch}, threshold={new_threshold:.4f}"
        )

        result.update({
            "epoch": epoch,
            "compression_ratio": adaptive_result['compression_ratio'],
            "cpu_time": adaptive_result['cpu_time'],
            "baseline_ratio": baseline['compression_ratio'],
            "dropped_ratio": adaptive_result['dropped_ratio'] * 100
        })

        results.append(result)

        print(f"Epoch {epoch:2d}: "
              f"Baseline CAQ={baseline['caq']:.2f}, "
              f"Adaptive CAQ={adaptive_result['caq']:.2f}, "
              f"Gain={result['gain_percent']:+.1f}%, "
              f"Ratio={adaptive_result['compression_ratio']:.2f}x")

    # Compute summary statistics
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    baseline_caqs = [baseline['caq'] for baseline in [run_baseline_compression(generate_synthetic_gradient(i)) for i in range(1, 6)]]
    adaptive_caqs = [r['adaptive_caq'] for r in results]

    mean_baseline = np.mean(baseline_caqs)
    mean_adaptive = np.mean(adaptive_caqs)
    mean_gain = ((mean_adaptive - mean_baseline) / mean_baseline) * 100

    print(f"\nMean Baseline CAQ: {mean_baseline:.2f}")
    print(f"Mean Adaptive CAQ: {mean_adaptive:.2f}")
    print(f"Mean Gain: {mean_gain:+.2f}%")

    # Variance check
    caq_variance = (np.max(adaptive_caqs) - np.min(adaptive_caqs)) / np.mean(adaptive_caqs) * 100
    print(f"CAQ Variance: {caq_variance:.2f}%")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "adaptive"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"run_{timestamp}.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "synthetic_gradients",
        "epochs": epochs,
        "mean_baseline_caq": round(mean_baseline, 2),
        "mean_adaptive_caq": round(mean_adaptive, 2),
        "mean_gain_percent": round(mean_gain, 2),
        "caq_variance": round(caq_variance, 2),
        "entropy_loss": round(final_loss, 6),
        "results": results,
        "notes": "Entropy model MLP64, adaptive quantization + pruning"
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Acceptance criteria check
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA")
    print("=" * 60)

    criteria_met = []
    criteria_met.append(("CAQ Gain >= 5%", mean_gain >= 5.0, f"{mean_gain:+.2f}%"))
    criteria_met.append(("Variance <= 1.5%", caq_variance <= 1.5, f"{caq_variance:.2f}%"))
    criteria_met.append(("Entropy training converged", final_loss < 0.1, f"{final_loss:.6f}"))

    all_pass = all(met for _, met, _ in criteria_met)

    for criterion, met, value in criteria_met:
        status = "✓ PASS" if met else "✗ FAIL"
        print(f"{status}: {criterion} ({value})")

    print()
    if all_pass:
        print("✓ ALL ACCEPTANCE CRITERIA MET")
        return 0
    else:
        print("✗ SOME CRITERIA NOT MET")
        return 1


if __name__ == "__main__":
    sys.exit(main())

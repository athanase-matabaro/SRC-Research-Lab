#!/usr/bin/env python3
"""
Unit tests for Adaptive Learned Compression Model (Phase H.3).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_model.neural_entropy import NeuralEntropyPredictor
from adaptive_model.gradient_encoder import GradientEncoder
from adaptive_model.scheduler import CompressionScheduler
from adaptive_model.utils import compute_tensor_stats, format_adaptive_result


class TestNeuralEntropy:
    """Test neural entropy predictor."""

    def test_init(self):
        """Test predictor initialization."""
        predictor = NeuralEntropyPredictor(hidden_size=32)
        assert predictor.hidden_size == 32
        assert predictor.W1.shape == (6, 32)
        assert predictor.W2.shape == (32, 1)

    def test_forward_pass(self):
        """Test forward propagation."""
        predictor = NeuralEntropyPredictor(hidden_size=16)
        features = np.random.randn(6)
        output = predictor.forward(features)
        assert 0.2 <= output <= 0.9  # min_bits to max_bits

    def test_train(self):
        """Test training loop."""
        np.random.seed(42)
        predictor = NeuralEntropyPredictor(hidden_size=16)

        features = [np.random.randn(6) for _ in range(20)]
        targets = [np.random.uniform(0.3, 0.8) for _ in range(20)]

        loss = predictor.train(features, targets, epochs=5, verbose=False)
        assert loss < 0.1  # Should converge reasonably

    def test_predict_entropy_map(self):
        """Test entropy map prediction."""
        predictor = NeuralEntropyPredictor()
        tensor = np.random.randn(10, 10)
        entropy_map = predictor.predict_entropy_map(tensor)
        assert entropy_map.shape == tensor.shape
        assert np.all((entropy_map >= 0.2) & (entropy_map <= 0.9))

    def test_save_load(self):
        """Test model save/load."""
        import tempfile
        predictor = NeuralEntropyPredictor()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name

        predictor.save_model(tmp_path)

        predictor2 = NeuralEntropyPredictor()
        predictor2.load_model(tmp_path)

        assert np.allclose(predictor.W1, predictor2.W1)
        Path(tmp_path).unlink()


class TestGradientEncoder:
    """Test gradient encoder."""

    def test_init(self):
        """Test encoder initialization."""
        encoder = GradientEncoder()
        assert encoder.base_precision == 8
        assert encoder.threshold == 0.01

    def test_quantize(self):
        """Test adaptive quantization."""
        encoder = GradientEncoder()
        tensor = np.random.randn(10, 10).astype(np.float32)
        entropy_map = np.full_like(tensor, 0.5)

        quantized, scales = encoder.adaptive_quantize(tensor, entropy_map)
        assert quantized.shape == tensor.shape
        assert scales.shape[0] == tensor.shape[0]

    def test_prune(self):
        """Test tensor pruning."""
        encoder = GradientEncoder()
        tensor = np.random.randn(100, 100).astype(np.float32)

        pruned, mask, ratio = encoder.prune_tensor(tensor, threshold=0.01)
        assert pruned.shape == tensor.shape
        assert mask.shape == tensor.shape
        assert 0 <= ratio <= encoder.max_drop_ratio

    def test_compress(self):
        """Test tensor compression."""
        encoder = GradientEncoder()
        tensor = np.random.randn(50, 50).astype(np.float32)

        result = encoder.compress_tensor(tensor)
        assert result['compression_ratio'] > 0
        assert result['caq'] > 0
        assert 'pruned' in result

    def test_decompress(self):
        """Test decompression."""
        encoder = GradientEncoder()
        tensor = np.random.randn(20, 20).astype(np.float32)

        compressed = encoder.compress_tensor(tensor)
        reconstructed = encoder.decompress_tensor(compressed)
        assert reconstructed.shape == tensor.shape

    def test_reconstruction_error(self):
        """Test reconstruction error computation."""
        encoder = GradientEncoder()
        original = np.random.randn(30, 30)
        reconstructed = original + np.random.randn(30, 30) * 0.1

        errors = encoder.compute_reconstruction_error(original, reconstructed)
        assert 'mse' in errors
        assert 'mae' in errors
        assert errors['mse'] > 0


class TestScheduler:
    """Test compression scheduler."""

    def test_init(self):
        """Test scheduler initialization."""
        scheduler = CompressionScheduler()
        assert scheduler.threshold == 0.01
        assert scheduler.epoch == 0

    def test_update_improving(self):
        """Test update with improving CAQ."""
        scheduler = CompressionScheduler()

        # Improving trend
        scheduler.update(4.0)
        scheduler.update(4.5)
        threshold = scheduler.update(5.0)

        # Should decrease (more aggressive)
        assert threshold < 0.01

    def test_update_declining(self):
        """Test update with declining CAQ."""
        scheduler = CompressionScheduler()

        # Declining trend
        scheduler.update(5.0)
        scheduler.update(4.5)
        threshold = scheduler.update(4.0)

        # Should increase (more conservative)
        assert threshold > 0.01

    def test_get_stats(self):
        """Test statistics retrieval."""
        scheduler = CompressionScheduler()
        scheduler.update(4.0)
        scheduler.update(4.5)

        stats = scheduler.get_stats()
        assert 'epoch' in stats
        assert 'mean_caq' in stats
        assert stats['epoch'] == 2

    def test_reset(self):
        """Test scheduler reset."""
        scheduler = CompressionScheduler()
        scheduler.update(4.0)
        scheduler.reset()

        assert scheduler.epoch == 0
        assert len(scheduler.caq_history) == 0


class TestUtils:
    """Test utility functions."""

    def test_compute_stats(self):
        """Test tensor statistics computation."""
        tensor = np.random.randn(50, 50)
        stats = compute_tensor_stats(tensor)

        assert 'mean' in stats
        assert 'var' in stats
        assert 'sparsity' in stats
        assert 0 <= stats['sparsity'] <= 1

    def test_format_result(self):
        """Test result formatting."""
        result = format_adaptive_result(
            status="PASS",
            baseline_caq=4.0,
            adaptive_caq=4.5,
            variance=0.5
        )

        assert result['status'] == 'PASS'
        assert result['gain_percent'] > 0
        assert 'baseline_caq' in result

    def test_format_with_loss(self):
        """Test formatting with entropy loss."""
        result = format_adaptive_result(
            status="PASS",
            baseline_caq=4.0,
            adaptive_caq=4.5,
            variance=1.0,
            entropy_loss=0.01
        )

        assert 'entropy_loss' in result
        assert result['entropy_loss'] == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

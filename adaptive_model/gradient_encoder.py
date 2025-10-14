#!/usr/bin/env python3
"""
Gradient Encoder with Adaptive Quantization and Pruning.
"""

import numpy as np
import tempfile
import time
from pathlib import Path
import yaml


class GradientEncoder:
    """Adaptive tensor encoder with gradient-aware quantization."""

    def __init__(self, config_path=None):
        self.base_precision = 8
        self.min_precision = 4
        self.threshold = 0.01
        self.max_drop_ratio = 0.25
        self.timeout = 300

        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.base_precision = config['quantization']['base_precision']
                self.min_precision = config['quantization']['min_precision']
                self.threshold = config['pruning']['threshold_init']
                self.max_drop_ratio = config['pruning']['max_drop_ratio']
                self.timeout = config['timeout_seconds']

    def adaptive_quantize(self, tensor, entropy_map):
        """Apply adaptive quantization based on entropy map."""
        scales = np.abs(np.mean(tensor, axis=tuple(range(1, tensor.ndim)), keepdims=True))
        scales = scales * entropy_map
        scales = np.maximum(scales, 1e-8)
        quantized = np.round(tensor / scales) * scales
        return quantized, scales

    def prune_tensor(self, tensor, threshold=None):
        """Prune small values from tensor."""
        if threshold is None:
            threshold = self.threshold

        abs_tensor = np.abs(tensor)
        auto_threshold = np.percentile(abs_tensor, 100 * threshold)
        mask = abs_tensor > auto_threshold
        pruned = tensor * mask
        dropped_ratio = 1.0 - (np.sum(mask) / mask.size)

        if dropped_ratio > self.max_drop_ratio:
            new_threshold = np.percentile(abs_tensor, 100 * self.max_drop_ratio)
            mask = abs_tensor > new_threshold
            pruned = tensor * mask
            dropped_ratio = 1.0 - (np.sum(mask) / mask.size)

        return pruned, mask, dropped_ratio

    def compress_tensor(self, tensor, entropy_map=None):
        """Compress tensor using adaptive quantization."""
        start_time = time.time()

        if entropy_map is None:
            entropy_map = np.full_like(tensor, 0.5, dtype=np.float32)

        # Adaptive quantization
        quantized, scales = self.adaptive_quantize(tensor, entropy_map)

        # Pruning
        pruned, mask, dropped_ratio = self.prune_tensor(quantized)

        # Backend compression
        original_size = tensor.nbytes

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name

        np.savez_compressed(tmp_path, tensor=pruned, mask=mask, scales=scales)
        compressed_size = Path(tmp_path).stat().st_size

        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        cpu_time = time.time() - start_time

        # Compute CAQ
        from metrics.caq_metric import compute_caq
        caq = compute_caq(compression_ratio, cpu_time)

        Path(tmp_path).unlink()

        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "cpu_time": cpu_time,
            "caq": caq,
            "dropped_ratio": dropped_ratio,
            "quantized": quantized,
            "pruned": pruned,
            "mask": mask,
            "scales": scales
        }

    def decompress_tensor(self, compressed_data):
        """Decompress tensor."""
        return compressed_data['pruned']

    def compute_reconstruction_error(self, original, reconstructed):
        """Compute reconstruction error metrics."""
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))

        original_norm = np.linalg.norm(original)
        if original_norm > 0:
            relative_error = np.linalg.norm(original - reconstructed) / original_norm
        else:
            relative_error = 0.0

        return {
            "mse": float(mse),
            "mae": float(mae),
            "relative_error": float(relative_error)
        }

"""
Adaptive Learned Compression Model (ALCM) - Phase H.3

Implements neural entropy modeling and gradient-aware compression
for efficient model checkpoint and tensor compression.

Offline, CPU-only, reproducible compression with CAQ optimization.
"""

__version__ = "1.0.0"

from adaptive_model.neural_entropy import NeuralEntropyPredictor
from adaptive_model.gradient_encoder import GradientEncoder  
from adaptive_model.scheduler import CompressionScheduler
from adaptive_model.utils import compute_tensor_stats, format_adaptive_result

__all__ = [
    "NeuralEntropyPredictor",
    "GradientEncoder",
    "CompressionScheduler",
    "compute_tensor_stats",
    "format_adaptive_result"
]

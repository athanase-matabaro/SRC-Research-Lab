#!/usr/bin/env python3
"""
Neural Entropy Predictor for Adaptive Compression.

Implements a lightweight MLP that predicts bit allocation from tensor statistics.
Offline training using cross-entropy loss against empirical compression ratios.
"""

import numpy as np
from pathlib import Path
import yaml


class NeuralEntropyPredictor:
    """
    2-layer MLP for predicting entropy-based bit allocation.

    Architecture: Input(6) -> Hidden(64) -> ReLU -> Output(1) -> Sigmoid
    """

    def __init__(self, config_path=None, hidden_size=64, learning_rate=0.001):
        """
        Initialize the neural entropy predictor.

        Args:
            config_path: Path to entropy_config.yaml
            hidden_size: Number of hidden units
            learning_rate: Learning rate for training
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Load config if provided
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.hidden_size = config['model']['hidden_size']
                self.learning_rate = config['model']['learning_rate']
                self.min_bits = config['scheduler']['min_bits']
                self.max_bits = config['scheduler']['max_bits']
        else:
            self.min_bits = 0.2
            self.max_bits = 0.9

        # Input features: mean, var, std, skew, kurtosis, sparsity
        self.input_size = 6
        self.output_size = 1

        # Initialize weights (Xavier initialization)
        self._init_weights()

        # Training history
        self.losses = []

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        # Layer 1: input -> hidden
        limit1 = np.sqrt(6.0 / (self.input_size + self.hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1,
                                    (self.input_size, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)

        # Layer 2: hidden -> output
        limit2 = np.sqrt(6.0 / (self.hidden_size + self.output_size))
        self.W2 = np.random.uniform(-limit2, limit2,
                                    (self.hidden_size, self.output_size))
        self.b2 = np.zeros(self.output_size)

    def _relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """ReLU derivative."""
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, x):
        """Sigmoid derivative."""
        s = self._sigmoid(x)
        return s * (1 - s)

    def forward(self, features):
        """
        Forward pass through the network.

        Args:
            features: Input features (6-dimensional)

        Returns:
            Predicted bit allocation [0, 1]
        """
        # Layer 1
        self.z1 = np.dot(features, self.W1) + self.b1
        self.a1 = self._relu(self.z1)

        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._sigmoid(self.z2)

        # Scale to [min_bits, max_bits]
        output = self.min_bits + (self.max_bits - self.min_bits) * self.a2

        return output[0]

    def backward(self, features, target, output):
        """
        Backward pass and weight update.

        Args:
            features: Input features
            target: Target bit allocation
            output: Network output
        """
        # Compute loss gradient
        # MSE loss: (output - target)^2
        dL_doutput = 2 * (output - target)

        # Scale back to [0, 1] for gradient
        dL_da2 = dL_doutput * (self.max_bits - self.min_bits)

        # Layer 2 gradients
        dL_dz2 = dL_da2 * self._sigmoid_derivative(self.z2)
        dL_dW2 = np.outer(self.a1, dL_dz2)
        dL_db2 = dL_dz2

        # Layer 1 gradients
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * self._relu_derivative(self.z1)
        dL_dW1 = np.outer(features, dL_dz1)
        dL_db1 = dL_dz1

        # Update weights
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1

    def train(self, feature_list, target_list, epochs=10, verbose=True):
        """
        Train the model on a dataset.

        Args:
            feature_list: List of feature vectors
            target_list: List of target bit allocations
            epochs: Number of training epochs
            verbose: Print training progress

        Returns:
            Final loss value
        """
        n_samples = len(feature_list)

        for epoch in range(epochs):
            total_loss = 0.0

            for features, target in zip(feature_list, target_list):
                # Forward pass
                output = self.forward(features)

                # Compute loss (MSE)
                loss = (output - target) ** 2
                total_loss += loss

                # Backward pass
                self.backward(features, target, output)

            avg_loss = total_loss / n_samples
            self.losses.append(avg_loss)

            if verbose and (epoch % 2 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        return self.losses[-1]

    def predict_entropy_map(self, tensor, stats=None):
        """
        Predict entropy map for a tensor.

        Args:
            tensor: NumPy array
            stats: Pre-computed statistics (optional)

        Returns:
            Entropy map with same shape as tensor
        """
        from adaptive_model.utils import compute_tensor_stats

        if stats is None:
            stats = compute_tensor_stats(tensor)

        # Create feature vector
        features = np.array([
            stats['mean'],
            stats['var'],
            stats['std'],
            stats['skew'],
            stats['kurtosis'],
            stats['sparsity']
        ])

        # Get prediction
        bit_allocation = self.forward(features)

        # Create entropy map (uniform for now, can be made spatial)
        entropy_map = np.full_like(tensor, bit_allocation, dtype=np.float32)

        return entropy_map

    def save_model(self, path):
        """Save model weights."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 hidden_size=self.hidden_size, min_bits=self.min_bits,
                 max_bits=self.max_bits)

    def load_model(self, path):
        """Load model weights."""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.hidden_size = int(data['hidden_size'])
        self.min_bits = float(data['min_bits'])
        self.max_bits = float(data['max_bits'])

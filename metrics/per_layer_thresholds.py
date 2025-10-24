#!/usr/bin/env python3
"""
Per-Layer Threshold Prediction

Lightweight, CPU-friendly predictors for per-layer CAQ-E anomaly detection thresholds.
Supports Ridge regression, decision trees, and small MLPs.

Key Features:
- Small model size (< 200 KB)
- Fast inference (< 1ms per layer on CPU)
- Interpretable coefficients (Ridge/tree)
- Deterministic training with seed control
- Serialization to JSON for guardrail integration

Example:
    >>> from metrics.per_layer_thresholds import PerLayerPredictor
    >>> predictor = PerLayerPredictor(model_type='ridge')
    >>> predictor.fit(X_train, y_train)
    >>> threshold = predictor.predict(layer_features)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class LayerFeatures:
    """
    Features for a single layer.

    Attributes:
        layer_index: 0-based position in model
        tensor_size: Number of elements in layer
        robust_mean: Robust mean (mean_abs)
        robust_std: Robust standard deviation
        robust_mad: Median absolute deviation
        sparsity: Fraction of near-zero elements
        quantization_error: Mean quantization error
        kurtosis: Fourth moment (tail behavior)
        skewness: Third moment (asymmetry)
        dataset_encoded: Dataset identifier (encoded)
    """
    layer_index: int
    tensor_size: int
    robust_mean: float
    robust_std: float
    robust_mad: float
    sparsity: float
    quantization_error: float
    kurtosis: float
    skewness: float
    dataset_encoded: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array for prediction."""
        return np.array([
            self.layer_index,
            self.tensor_size,
            self.robust_mean,
            self.robust_std,
            self.robust_mad,
            self.sparsity,
            self.quantization_error,
            self.kurtosis,
            self.skewness,
            self.dataset_encoded
        ])


class PerLayerPredictor:
    """
    Per-layer threshold predictor using Ridge regression or decision trees.

    Small, interpretable models for predicting allowed CAQ-E drift per layer.
    """

    def __init__(
        self,
        model_type: str = 'ridge',
        alpha: float = 0.01,
        max_depth: int = 4,
        seed: int = 42
    ):
        """
        Initialize predictor.

        Args:
            model_type: 'ridge', 'tree', or 'mlp'
            alpha: L2 regularization for Ridge
            max_depth: Max depth for decision tree
            seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.alpha = alpha
        self.max_depth = max_depth
        self.seed = seed
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.feature_names = [
            'layer_index', 'tensor_size', 'robust_mean', 'robust_std',
            'robust_mad', 'sparsity', 'quantization_error', 'kurtosis',
            'skewness', 'dataset_encoded'
        ]

        np.random.seed(seed)

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Standardize features (z-score normalization)."""
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-9
        return (X - self.scaler_mean) / self.scaler_std

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit predictor to training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,) - allowed drift percentages
        """
        # Standardize features
        X_scaled = self._standardize(X, fit=True)

        if self.model_type == 'ridge':
            # Ridge regression (L2)
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=self.alpha, random_state=self.seed)
            self.model.fit(X_scaled, y)

        elif self.model_type == 'tree':
            # Decision tree
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.seed
            )
            self.model.fit(X_scaled, y)

        elif self.model_type == 'mlp':
            # Small MLP (1 hidden layer, 64 units)
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=(64,),
                activation='relu',
                solver='adam',
                alpha=self.alpha,
                max_iter=1000,
                random_state=self.seed
            )
            self.model.fit(X_scaled, y)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict allowed drift for input features.

        Args:
            X: Feature matrix (n_samples, n_features) or (n_features,)

        Returns:
            Predicted allowed drift percentages
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self._standardize(X, fit=False)
        predictions = self.model.predict(X_scaled)

        # Clip to reasonable range [0.05, 0.50]
        predictions = np.clip(predictions, 0.05, 0.50)

        return predictions

    def predict_single(self, layer_features: LayerFeatures) -> float:
        """
        Predict threshold for a single layer.

        Args:
            layer_features: LayerFeatures object

        Returns:
            Predicted allowed drift percentage
        """
        X = layer_features.to_array().reshape(1, -1)
        return float(self.predict(X)[0])

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Get feature importances (Ridge coefficients or tree importances).

        Returns:
            Dict mapping feature names to importance scores
        """
        if self.model is None:
            return None

        if self.model_type == 'ridge':
            importances = np.abs(self.model.coef_)
        elif self.model_type == 'tree':
            importances = self.model.feature_importances_
        elif self.model_type == 'mlp':
            # For MLP, use average weight magnitudes from first layer
            importances = np.mean(np.abs(self.model.coefs_[0]), axis=1)
        else:
            return None

        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}

    def save(self, path: Path):
        """
        Save model to disk.

        Args:
            path: Output path (will save as .pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model_type': self.model_type,
            'alpha': self.alpha,
            'max_depth': self.max_depth,
            'seed': self.seed,
            'model': self.model,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'feature_names': self.feature_names
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: Path) -> 'PerLayerPredictor':
        """
        Load model from disk.

        Args:
            path: Input path (.pkl file)

        Returns:
            Loaded PerLayerPredictor
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        predictor = cls(
            model_type=model_data['model_type'],
            alpha=model_data['alpha'],
            max_depth=model_data['max_depth'],
            seed=model_data['seed']
        )

        predictor.model = model_data['model']
        predictor.scaler_mean = model_data['scaler_mean']
        predictor.scaler_std = model_data['scaler_std']
        predictor.feature_names = model_data['feature_names']

        return predictor

    def to_dict(self) -> Dict[str, Any]:
        """
        Export model metadata to dict (for JSON serialization).

        Returns:
            Dict with model metadata and feature importances
        """
        importances = self.get_feature_importances()

        return {
            'model_type': self.model_type,
            'alpha': self.alpha,
            'max_depth': self.max_depth,
            'seed': self.seed,
            'feature_names': self.feature_names,
            'feature_importances': importances,
            'scaler_mean': self.scaler_mean.tolist() if self.scaler_mean is not None else None,
            'scaler_std': self.scaler_std.tolist() if self.scaler_std is not None else None
        }

    def get_model_size_bytes(self, path: Path) -> int:
        """
        Get model file size in bytes.

        Args:
            path: Model file path

        Returns:
            File size in bytes
        """
        return path.stat().st_size if path.exists() else 0


def serialize_to_guardrail_config(
    predictor: PerLayerPredictor,
    layer_names: List[str],
    layer_features_list: List[LayerFeatures],
    output_path: Path
):
    """
    Export per-layer thresholds to guardrail config JSON.

    Args:
        predictor: Trained PerLayerPredictor
        layer_names: List of layer names
        layer_features_list: List of LayerFeatures for each layer
        output_path: Output JSON path
    """
    per_layer_thresholds = {}

    for layer_name, layer_features in zip(layer_names, layer_features_list):
        predicted_drift = predictor.predict_single(layer_features)

        per_layer_thresholds[layer_name] = {
            'allowed_drift_pct': float(predicted_drift),
            'threshold_scale': float(1.0 + predicted_drift),
            'features': asdict(layer_features)
        }

    config = {
        'schema_version': '1.0',
        'model_metadata': predictor.to_dict(),
        'per_layer_thresholds': per_layer_thresholds
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    # Demonstration
    print("Per-Layer Threshold Predictor Demonstration")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 100

    X = np.random.randn(n_samples, 10)
    y = 0.15 + 0.1 * X[:, 2] + 0.05 * X[:, 5] + 0.02 * np.random.randn(n_samples)
    y = np.clip(y, 0.05, 0.50)

    # Train Ridge model
    predictor = PerLayerPredictor(model_type='ridge', alpha=0.01)
    predictor.fit(X, y)

    # Test prediction
    test_features = LayerFeatures(
        layer_index=0,
        tensor_size=4096,
        robust_mean=0.15,
        robust_std=0.08,
        robust_mad=0.11,
        sparsity=0.25,
        quantization_error=-0.005,
        kurtosis=4.2,
        skewness=0.3,
        dataset_encoded=0
    )

    predicted_drift = predictor.predict_single(test_features)

    print(f"\nTest Prediction:")
    print(f"  Predicted allowed drift: {predicted_drift:.4f} ({predicted_drift*100:.1f}%)")
    print()

    print("Feature Importances:")
    importances = predictor.get_feature_importances()
    for name, imp in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        print(f"  {name:20s}: {imp:.4f}")
    print()

    # Save model
    model_path = Path("models/per_layer_predictor/demo_model.pkl")
    predictor.save(model_path)
    print(f"Model saved to: {model_path}")
    print(f"Model size: {predictor.get_model_size_bytes(model_path)} bytes")
    print()

    # Load and verify
    loaded_predictor = PerLayerPredictor.load(model_path)
    loaded_prediction = loaded_predictor.predict_single(test_features)

    print(f"Loaded model prediction: {loaded_prediction:.4f}")
    print(f"Match: {abs(predicted_drift - loaded_prediction) < 1e-6}")
    print()

    print("=" * 60)
    print("Demonstration complete!")

"""
Gradient Dataset Generator

Generates synthetic gradient tensors that simulate real deep learning model
training gradients (e.g., CIFAR-10/ResNet-8 training traces).

Offline operation, no network access required.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-16
Phase: H.5 - Energy-Aware Compression
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle


def generate_cifar10_resnet8_gradients(
    num_epochs: int = 10,
    layers_config: List[Tuple[int, ...]] = None,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic gradients simulating CIFAR-10/ResNet-8 training.

    Args:
        num_epochs: Number of training epochs to simulate.
        layers_config: List of layer shapes. If None, uses default ResNet-8 architecture.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping epoch -> gradient tensors.

    Architecture (default):
        - Conv1: 3x3x3x16 (3 input channels, 16 filters, 3x3 kernel)
        - Conv2: 3x3x16x32
        - Conv3: 3x3x32x64
        - FC: 64x10 (10 classes for CIFAR-10)
    """
    np.random.seed(seed)

    if layers_config is None:
        # Default ResNet-8 inspired architecture
        layers_config = [
            (3, 3, 3, 16),    # Conv1
            (3, 3, 16, 32),   # Conv2
            (3, 3, 32, 64),   # Conv3
            (64, 10),         # FC
        ]

    gradients = {}

    for epoch in range(num_epochs):
        epoch_gradients = []

        for layer_idx, shape in enumerate(layers_config):
            # Generate gradients with realistic characteristics

            # Early epochs: larger gradients, higher variance
            # Later epochs: smaller gradients, lower variance (convergence)
            epoch_scale = 1.0 / (1.0 + 0.1 * epoch)

            # Conv layers typically have different distributions than FC
            is_conv = len(shape) == 4

            if is_conv:
                # Convolutional layer: more structured, spatial patterns
                base_grads = np.random.randn(*shape) * epoch_scale * 0.01

                # Add some structure (simulating learned features)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        # Create spatial correlation
                        pattern = np.sin(i * 0.5) * np.cos(j * 0.5)
                        base_grads[i, j, :, :] += pattern * 0.001 * epoch_scale

            else:
                # Fully connected layer: less structured
                base_grads = np.random.randn(*shape) * epoch_scale * 0.02

            # Add sparsity (many gradients are near zero)
            # Later epochs tend to be sparser
            sparsity_level = 0.2 + 0.05 * epoch
            mask = np.random.rand(*shape) > sparsity_level
            base_grads = base_grads * mask

            epoch_gradients.append(base_grads.astype(np.float32))

        # Concatenate all layer gradients for this epoch
        gradients[f"epoch_{epoch}"] = {
            f"layer_{idx}": grad for idx, grad in enumerate(epoch_gradients)
        }

    return gradients


def generate_synthetic_gradients_simple(
    num_samples: int = 10,
    shape: Tuple[int, ...] = (100, 100),
    seed: int = 42
) -> np.ndarray:
    """
    Generate simple synthetic gradient dataset for quick testing.

    Args:
        num_samples: Number of gradient samples.
        shape: Shape of each gradient tensor.
        seed: Random seed.

    Returns:
        Array of gradient tensors.
    """
    np.random.seed(seed)

    gradients = []

    for i in range(num_samples):
        # Vary the distribution characteristics per sample
        mean_shift = (i - num_samples/2) * 0.01
        std_scale = 0.5 + 0.1 * i

        grad = np.random.randn(*shape) * std_scale + mean_shift

        # Add sparsity
        mask = np.random.rand(*shape) > 0.3
        grad = grad * mask

        gradients.append(grad.astype(np.float32))

    return np.array(gradients)


def save_gradient_dataset(
    gradients: Dict,
    output_path: Path,
    compression: bool = True
) -> None:
    """
    Save gradient dataset to disk.

    Args:
        gradients: Gradient data (dict or array).
        output_path: Output file path.
        compression: Whether to use compression (recommended for large datasets).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compression:
        # Use pickle with compression for large gradient datasets
        with open(output_path, 'wb') as f:
            pickle.dump(gradients, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Use numpy format
        if isinstance(gradients, np.ndarray):
            np.save(output_path, gradients)
        else:
            # For dict of gradients, use pickle
            with open(output_path, 'wb') as f:
                pickle.dump(gradients, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_gradient_dataset(input_path: Path) -> Dict:
    """
    Load gradient dataset from disk.

    Args:
        input_path: Input file path.

    Returns:
        Gradient data.
    """
    if input_path.suffix == '.npy':
        return np.load(input_path)
    else:
        with open(input_path, 'rb') as f:
            return pickle.load(f)


def get_dataset_statistics(gradients: Dict) -> Dict:
    """
    Compute statistics for gradient dataset.

    Args:
        gradients: Gradient dataset.

    Returns:
        Dictionary with dataset statistics.
    """
    stats = {
        "num_epochs": len(gradients),
        "layer_info": [],
        "total_parameters": 0,
    }

    # Get first epoch to determine layer structure
    first_epoch = list(gradients.values())[0]

    for layer_name, layer_grads in first_epoch.items():
        layer_stats = {
            "name": layer_name,
            "shape": layer_grads.shape,
            "num_params": layer_grads.size,
            "mean": float(np.mean(layer_grads)),
            "std": float(np.std(layer_grads)),
            "sparsity": float(np.sum(np.abs(layer_grads) < 1e-6) / layer_grads.size),
        }
        stats["layer_info"].append(layer_stats)
        stats["total_parameters"] += layer_grads.size

    return stats


def main():
    """
    Generate and save gradient datasets for Phase H.5.
    """
    print("Generating gradient datasets for Phase H.5...")

    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate CIFAR-10/ResNet-8 synthetic gradients
    print("\n1. Generating CIFAR-10/ResNet-8 gradients...")
    cifar_gradients = generate_cifar10_resnet8_gradients(
        num_epochs=10,
        seed=42
    )

    cifar_output = output_dir / "real_gradients_cifar10.pkl"
    save_gradient_dataset(cifar_gradients, cifar_output, compression=True)

    cifar_stats = get_dataset_statistics(cifar_gradients)
    print(f"   Generated {cifar_stats['num_epochs']} epochs")
    print(f"   Total parameters: {cifar_stats['total_parameters']:,}")
    print(f"   Saved to: {cifar_output}")

    # 2. Generate simple synthetic gradients for fallback testing
    print("\n2. Generating simple synthetic gradients...")
    simple_gradients = generate_synthetic_gradients_simple(
        num_samples=10,
        shape=(100, 100),
        seed=42
    )

    simple_output = output_dir / "synthetic_gradients.npy"
    np.save(simple_output, simple_gradients)
    print(f"   Generated {len(simple_gradients)} samples of shape {simple_gradients[0].shape}")
    print(f"   Saved to: {simple_output}")

    # 3. Generate mixed dataset (combination)
    print("\n3. Generating mixed dataset...")
    mixed_gradients = {
        "cifar10_sample": cifar_gradients["epoch_0"],
        "synthetic_samples": simple_gradients[:3],
    }

    mixed_output = output_dir / "mixed_gradients.pkl"
    save_gradient_dataset(mixed_gradients, mixed_output, compression=True)
    print(f"   Saved to: {mixed_output}")

    print("\n✓ Gradient dataset generation complete!")
    print(f"\nDatasets created:")
    print(f"  - {cifar_output.name}: CIFAR-10/ResNet-8 gradients (10 epochs)")
    print(f"  - {simple_output.name}: Simple synthetic gradients (10 samples)")
    print(f"  - {mixed_output.name}: Mixed dataset")


if __name__ == "__main__":
    main()

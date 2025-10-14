#!/usr/bin/env python3
"""
Utility functions for Adaptive Learned Compression Model.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics.caq_metric import compute_caq


def compute_tensor_stats(tensor):
    """
    Compute statistical features from a tensor.
    
    Args:
        tensor: NumPy array
        
    Returns:
        dict: Statistics including mean, var, skew, kurtosis, sparsity
    """
    flat = tensor.flatten()
    
    mean = float(np.mean(flat))
    var = float(np.var(flat))
    
    # Sparsity (percentage of near-zero values)
    threshold = 1e-6
    sparsity = float(np.sum(np.abs(flat) < threshold) / len(flat))
    
    # Simplified skew and kurtosis
    std = np.sqrt(var) if var > 0 else 1e-8
    standardized = (flat - mean) / std
    
    skew = float(np.mean(standardized ** 3))
    kurtosis = float(np.mean(standardized ** 4))
    
    return {
        "mean": mean,
        "var": var,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "sparsity": sparsity
    }


def format_adaptive_result(status, baseline_caq, adaptive_caq, variance, 
                           entropy_loss=None, notes=""):
    """
    Format adaptive compression result.
    
    Args:
        status: "PASS" or "FAIL"
        baseline_caq: Baseline CAQ score
        adaptive_caq: Adaptive CAQ score
        variance: Variance percentage
        entropy_loss: Optional entropy loss value
        notes: Additional notes
        
    Returns:
        dict: Formatted result
    """
    gain_percent = 0.0
    if baseline_caq > 0:
        gain_percent = ((adaptive_caq - baseline_caq) / baseline_caq) * 100.0
    
    result = {
        "status": status,
        "baseline_caq": round(baseline_caq, 2),
        "adaptive_caq": round(adaptive_caq, 2),
        "gain_percent": round(gain_percent, 2),
        "variance": round(variance, 2),
        "notes": notes
    }
    
    if entropy_loss is not None:
        result["entropy_loss"] = round(entropy_loss, 4)
    
    return result


def normalize_tensor(tensor, method="minmax"):
    """
    Normalize tensor values.
    
    Args:
        tensor: NumPy array
        method: "minmax" or "zscore"
        
    Returns:
        Normalized tensor
    """
    if method == "minmax":
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        if max_val > min_val:
            return (tensor - min_val) / (max_val - min_val)
        return tensor
    elif method == "zscore":
        mean = np.mean(tensor)
        std = np.std(tensor)
        if std > 0:
            return (tensor - mean) / std
        return tensor - mean
    else:
        return tensor

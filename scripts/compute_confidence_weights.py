#!/usr/bin/env python3
"""
Compute Confidence Weights from Uncertainties

Utility to transform σ → weights for batch calibration.

Phase D.4 - Adaptive Confidence-Weighted Guardrails
"""

import numpy as np
import json
import argparse
from pathlib import Path


def compute_weights(uncertainties, method="exponential", lambda_param=1.0):
    """Compute weights from uncertainties."""
    sigmas = [u["std"] for u in uncertainties.values()]
    sigma_ref = np.median(sigmas)
    
    weights = {}
    for layer, unc in uncertainties.items():
        sigma = unc["std"]
        if method == "exponential":
            w = np.exp(-lambda_param * sigma / sigma_ref)
        else:  # inverse_variance
            w = 1.0 / (sigma ** 2 + 0.01)
        weights[layer] = float(w)
    
    # Normalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="exponential")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_param")
    args = parser.parse_args()
    
    with open(args.uncertainty) as f:
        unc = json.load(f)
    
    weights = compute_weights(unc, args.method, args.lambda_param)
    
    with open(args.output, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"Computed weights: {weights}")


if __name__ == "__main__":
    main()

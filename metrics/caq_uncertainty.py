#!/usr/bin/env python3
"""
CAQ-E Uncertainty Quantification

Statistically grounded uncertainty estimation for CAQ-E (Compression-Aware Quality per Energy)
metrics. Provides multiple estimation methods (bootstrap, Bayesian, MAD-based) and adaptive
mode selection based on data characteristics.

Phase D.3 - Uncertainty Quantification & Confidence Propagation
Builds on Phase D.1 (robust estimators) and D.2 (per-layer thresholds).

Features:
- Bootstrap resampling for empirical confidence intervals
- Bayesian inference with Normal-Gamma prior
- MAD-based empirical confidence intervals
- Adaptive estimator selection based on skewness and kurtosis
- Deterministic RNG (seed control)
- Offline-only execution

Mathematical Background:
- Bootstrap: Efron (1979) - Resampling-based CI estimation
- Bayesian: Normal-Gamma conjugate prior for mean-variance estimation
- MAD: Median Absolute Deviation as robust scale estimator
- Law of error propagation for uncertainty propagation across layers

Author: Phase D.3 Implementation
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings


class EstimationMode(Enum):
    """Uncertainty estimation methods."""
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"
    MAD_CONF = "mad_conf"
    AUTO = "auto"


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty quantification results."""
    mean: float
    median: float
    variance: float
    std: float
    ci_lower: float  # Lower bound of confidence interval
    ci_upper: float  # Upper bound of confidence interval
    ci_level: float  # Confidence level (e.g., 0.95)
    method: str  # Estimation method used
    n_samples: int  # Number of samples used

    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty (CI width / mean)."""
        if abs(self.mean) < 1e-10:
            return np.inf
        return self.ci_width / abs(self.mean)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'mean': float(self.mean),
            'median': float(self.median),
            'variance': float(self.variance),
            'std': float(self.std),
            'ci_lower': float(self.ci_lower),
            'ci_upper': float(self.ci_upper),
            'ci_level': float(self.ci_level),
            'ci_width': float(self.ci_width),
            'relative_uncertainty': float(self.relative_uncertainty),
            'method': self.method,
            'n_samples': int(self.n_samples)
        }


class CAQUncertaintyEstimator:
    """
    Uncertainty quantification for CAQ-E metrics.

    Supports multiple estimation methods:
    - bootstrap: Resampling-based confidence intervals
    - bayesian: MAP estimate with posterior variance
    - mad_conf: MAD-based empirical confidence intervals
    - auto: Adaptive method selection based on data characteristics

    All methods are deterministic given a fixed random seed.
    """

    def __init__(
        self,
        method: Union[EstimationMode, str] = EstimationMode.AUTO,
        ci_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42
    ):
        """
        Initialize uncertainty estimator.

        Args:
            method: Estimation method to use
            ci_level: Confidence interval level (default 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples (for bootstrap method)
            seed: Random seed for deterministic behavior
        """
        if isinstance(method, str):
            method = EstimationMode(method)

        self.method = method
        self.ci_level = ci_level
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Validate parameters
        if not 0 < ci_level < 1:
            raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
        if n_bootstrap < 100:
            warnings.warn(f"n_bootstrap={n_bootstrap} is low, recommend >= 1000")

    def estimate(
        self,
        data: np.ndarray,
        method: Optional[Union[EstimationMode, str]] = None
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty for given data.

        Args:
            data: 1D array of metric values
            method: Override default estimation method (optional)

        Returns:
            UncertaintyEstimate object with mean, variance, and confidence intervals
        """
        data = np.asarray(data).flatten()

        if len(data) == 0:
            raise ValueError("Cannot estimate uncertainty from empty data")

        # Determine method
        est_method = method if method is not None else self.method
        if isinstance(est_method, str):
            est_method = EstimationMode(est_method)

        # Auto-select method if requested
        if est_method == EstimationMode.AUTO:
            est_method = self._select_method(data)

        # Dispatch to appropriate estimator
        if est_method == EstimationMode.BOOTSTRAP:
            return self._estimate_bootstrap(data)
        elif est_method == EstimationMode.BAYESIAN:
            return self._estimate_bayesian(data)
        elif est_method == EstimationMode.MAD_CONF:
            return self._estimate_mad_conf(data)
        else:
            raise ValueError(f"Unknown estimation method: {est_method}")

    def _select_method(self, data: np.ndarray) -> EstimationMode:
        """
        Automatically select estimation method based on data characteristics.

        Selection logic:
        - If n < 30: Use MAD (robust to small samples)
        - If |skewness| > 2 or kurtosis > 7: Use bootstrap (handles non-normal)
        - Otherwise: Use Bayesian (efficient for normal-like data)

        Args:
            data: 1D array of metric values

        Returns:
            Selected EstimationMode
        """
        n = len(data)

        # Small sample: use MAD
        if n < 30:
            return EstimationMode.MAD_CONF

        # Compute skewness and kurtosis
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std < 1e-10:
            # Zero variance: use MAD
            return EstimationMode.MAD_CONF

        # Standardized moments
        z = (data - mean) / std
        skewness = np.mean(z**3)
        kurtosis = np.mean(z**4)

        # Non-normal data: use bootstrap
        if abs(skewness) > 2.0 or kurtosis > 7.0:
            return EstimationMode.BOOTSTRAP

        # Normal-like data: use Bayesian
        return EstimationMode.BAYESIAN

    def _estimate_bootstrap(self, data: np.ndarray) -> UncertaintyEstimate:
        """
        Bootstrap resampling for confidence intervals.

        Uses percentile method with n_bootstrap resamples.
        Deterministic given fixed random seed.

        Args:
            data: 1D array of metric values

        Returns:
            UncertaintyEstimate with bootstrap-based confidence intervals
        """
        n = len(data)

        # Generate bootstrap samples
        bootstrap_means = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            # Resample with replacement
            indices = self.rng.randint(0, n, size=n)
            bootstrap_sample = data[indices]
            bootstrap_means[i] = np.mean(bootstrap_sample)

        # Compute statistics
        mean = np.mean(data)
        median = np.median(data)
        variance = np.var(data, ddof=1)
        std = np.sqrt(variance)

        # Percentile-based confidence interval
        alpha = (1 - self.ci_level) / 2
        ci_lower = np.percentile(bootstrap_means, 100 * alpha)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha))

        return UncertaintyEstimate(
            mean=mean,
            median=median,
            variance=variance,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method="bootstrap",
            n_samples=n
        )

    def _estimate_bayesian(self, data: np.ndarray) -> UncertaintyEstimate:
        """
        Bayesian inference with Normal-Gamma conjugate prior.

        Assumes data ~ Normal(mu, sigma^2) with conjugate prior:
        - mu | sigma^2 ~ Normal(mu0, sigma^2 / kappa0)
        - sigma^2 ~ InverseGamma(alpha0, beta0)

        Uses non-informative prior (mu0=0, kappa0=0, alpha0=0, beta0=0).

        Args:
            data: 1D array of metric values

        Returns:
            UncertaintyEstimate with Bayesian posterior-based confidence intervals
        """
        n = len(data)

        # Sample statistics
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # Non-informative prior parameters
        mu0, kappa0, alpha0, beta0 = 0.0, 0.0, 0.0, 0.0

        # Posterior parameters (Normal-Gamma conjugate update)
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * sample_mean) / kappa_n
        alpha_n = alpha0 + n / 2
        beta_n = beta0 + 0.5 * np.sum((data - sample_mean)**2) + \
                 (kappa0 * n * (sample_mean - mu0)**2) / (2 * kappa_n)

        # Posterior mean and variance for mu
        # E[mu | data] = mu_n
        # Var[mu | data] = beta_n / (alpha_n * kappa_n)
        posterior_mean = mu_n
        posterior_var = beta_n / (alpha_n * kappa_n) if alpha_n > 0 else sample_var / n

        # Confidence interval using Student's t-distribution
        # (mu - mu_n) / sqrt(posterior_var) ~ t_{2*alpha_n}
        from scipy import stats
        df = 2 * alpha_n
        t_critical = stats.t.ppf((1 + self.ci_level) / 2, df) if df > 0 else 1.96
        margin = t_critical * np.sqrt(posterior_var)

        ci_lower = posterior_mean - margin
        ci_upper = posterior_mean + margin

        return UncertaintyEstimate(
            mean=sample_mean,
            median=np.median(data),
            variance=sample_var,
            std=np.sqrt(sample_var),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method="bayesian",
            n_samples=n
        )

    def _estimate_mad_conf(self, data: np.ndarray) -> UncertaintyEstimate:
        """
        MAD-based empirical confidence intervals.

        Uses Median Absolute Deviation (MAD) as robust scale estimator.
        For normal data, MAD * 1.4826 ≈ standard deviation.

        CI constructed as: median ± k * MAD / sqrt(n)
        where k is chosen to achieve desired confidence level.

        Args:
            data: 1D array of metric values

        Returns:
            UncertaintyEstimate with MAD-based confidence intervals
        """
        n = len(data)

        # Robust statistics
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        # MAD to standard deviation (for normal data)
        robust_std = mad * 1.4826

        # Sample statistics for comparison
        mean = np.mean(data)
        variance = np.var(data, ddof=1)
        std = np.sqrt(variance)

        # Confidence interval using MAD
        # Use normal approximation: k * MAD / sqrt(n)
        from scipy import stats
        z_critical = stats.norm.ppf((1 + self.ci_level) / 2)
        margin = z_critical * robust_std / np.sqrt(n)

        ci_lower = median - margin
        ci_upper = median + margin

        return UncertaintyEstimate(
            mean=mean,
            median=median,
            variance=variance,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method="mad_conf",
            n_samples=n
        )


def estimate_uncertainty_batch(
    data_dict: Dict[str, np.ndarray],
    method: Union[EstimationMode, str] = EstimationMode.AUTO,
    ci_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, UncertaintyEstimate]:
    """
    Estimate uncertainty for multiple datasets (e.g., per-layer metrics).

    Args:
        data_dict: Dictionary mapping layer names to metric arrays
        method: Estimation method to use
        ci_level: Confidence interval level
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for deterministic behavior

    Returns:
        Dictionary mapping layer names to UncertaintyEstimate objects
    """
    estimator = CAQUncertaintyEstimator(
        method=method,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        seed=seed
    )

    results = {}
    for layer_name, data in data_dict.items():
        results[layer_name] = estimator.estimate(data)

    return results


def propagate_uncertainty(
    estimates: List[UncertaintyEstimate],
    operation: str = "sum"
) -> UncertaintyEstimate:
    """
    Propagate uncertainty through mathematical operations.

    Uses law of error propagation for independent variables:
    - Sum: Var(X + Y) = Var(X) + Var(Y)
    - Product: Var(XY) ≈ (X*Var(Y) + Y*Var(X)) for small relative errors

    Args:
        estimates: List of UncertaintyEstimate objects
        operation: Operation to propagate ("sum" or "product")

    Returns:
        Combined UncertaintyEstimate
    """
    if not estimates:
        raise ValueError("Cannot propagate uncertainty from empty list")

    if operation == "sum":
        # Sum of means
        combined_mean = sum(est.mean for est in estimates)
        combined_median = sum(est.median for est in estimates)

        # Sum of variances (assuming independence)
        combined_variance = sum(est.variance for est in estimates)
        combined_std = np.sqrt(combined_variance)

        # Conservative CI propagation
        ci_lower = sum(est.ci_lower for est in estimates)
        ci_upper = sum(est.ci_upper for est in estimates)

    elif operation == "product":
        # Product of means
        combined_mean = np.prod([est.mean for est in estimates])
        combined_median = np.prod([est.median for est in estimates])

        # Approximate variance for product (first-order Taylor)
        combined_variance = 0.0
        means = np.array([est.mean for est in estimates])
        variances = np.array([est.variance for est in estimates])

        for i in range(len(estimates)):
            # Partial derivative: ∂(∏X_i)/∂X_i = ∏_{j≠i} X_j
            partial = np.prod(np.delete(means, i))
            combined_variance += (partial ** 2) * variances[i]

        combined_std = np.sqrt(combined_variance)

        # Approximate CI
        ci_lower = combined_mean - 1.96 * combined_std
        ci_upper = combined_mean + 1.96 * combined_std

    else:
        raise ValueError(f"Unknown operation: {operation}")

    # Average confidence level and method
    avg_ci_level = np.mean([est.ci_level for est in estimates])
    methods = [est.method for est in estimates]
    combined_method = f"propagated_{operation}_{','.join(set(methods))}"

    total_samples = sum(est.n_samples for est in estimates)

    return UncertaintyEstimate(
        mean=combined_mean,
        median=combined_median,
        variance=combined_variance,
        std=combined_std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=avg_ci_level,
        method=combined_method,
        n_samples=total_samples
    )

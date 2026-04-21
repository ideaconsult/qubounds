import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from typing import Dict


# ============================================================================
# QUANTILE BIN FUNCTIONS
# ============================================================================

def compute_quantile_bins(values: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """
    Compute quantile bin edges from array of values
    
    Args:
        values: Array of numerical values
        n_bins: Number of bins (default 20)
    
    Returns:
        Array of n_bins+1 bin edges (e.g., 21 edges for 20 bins)
    
    Example:
        >>> values = np.array([1.2, 2.3, 1.8, 3.5, 2.1])
        >>> edges = compute_quantile_bins(values, n_bins=20)
        >>> print(edges.shape)  # (21,)
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(values, percentiles)
    return bin_edges


def discretize_interval(lower: float,
                        upper: float,
                        bin_edges: np.ndarray,
                        coverage: float = 0.90) -> np.ndarray:
    """
    Discretize a conformal prediction interval into bins using Gaussian approximation
    
    Args:
        lower: Lower bound of interval
        upper: Upper bound of interval
        bin_edges: Array of bin edges (length n_bins+1)
        coverage: Coverage level (default 0.90)
    
    Returns:
        Array of probabilities (length n_bins)
    """
    from scipy.stats import norm
    
    z_score = norm.ppf((1 + coverage) / 2)
    mu = (lower + upper) / 2
    sigma = (upper - lower) / (2 * z_score)
    
    return discretize_gaussian(mu, sigma, bin_edges)


def discretize_gaussian(
        mu: float, sigma: float, bin_edges: np.ndarray
        ) -> np.ndarray:
    """
    Discretize a Gaussian distribution into bins
    
    Args:
        mu: Mean of Gaussian
        sigma: Standard deviation of Gaussian
        bin_edges: Array of bin edges (length n_bins+1)
    
    Returns:
        Array of probabilities (length n_bins)
    
    Example:
        >>> mu, sigma = 2.5, 0.5
        >>> edges = np.array([0, 1, 2, 3, 4, 5])
        >>> probs = discretize_gaussian(mu, sigma, edges)
        >>> print(probs.sum())  # Should be close to 1.0
    """
    n_bins = len(bin_edges) - 1
    bins = np.zeros(n_bins)
    
    for i in range(n_bins):
        p_lower = norm.cdf(bin_edges[i], loc=mu, scale=sigma)
        p_upper = norm.cdf(bin_edges[i+1], loc=mu, scale=sigma)
        bins[i] = p_upper - p_lower
    
    return bins
#!/usr/bin/env python3
"""K-means"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """best n¹ clusters for a GMM using Bayesian Information Criterion"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax < 1:
        return None, None, None, None
    if kmax - kmin < 1:
        return None, None, None, None

    n, d = X.shape
    x = []
    y = []
    results = []
    n_cluster = np.arange(kmin, kmax + 1)
    for k in range(kmin, kmax + 1):
        pi, m, S, g, L = expectation_maximization(
            X, k, iterations, tol, verbose)
        results.append((pi, m, S))
        p = k * d + (k - 1) + k * d * (d + 1) / 2
        x.append(p * np.log(X.shape[0]) - 2 * L)
        y.append(L)
    b = np.array(x)
    likelihood = np.array(y)
    idx = np.argmin(b)
    best_k = n_cluster[idx]
    best_result = results[idx]
    return best_k, best_result, likelihood, b

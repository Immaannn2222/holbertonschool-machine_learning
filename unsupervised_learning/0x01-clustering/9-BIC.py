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
    b_lst = []
    l_lst = []
    results = []
    ks = []
    for k in range(kmin, kmax + 1):
        ks.append(k)
        em = expectation_maximization(X, k, iterations, tol, verbose)
        pi, m, S, g, L = em
        results.append((pi, m, S))
        p = k * d + (k - 1) + k * d * (d + 1) / 2
        b_lst.append(p * np.log(X.shape[0]) - 2 * L)
        l_lst.append(L)

    bics = np.array(b_lst)
    liklihoods = np.array(l_lst)
    best_idx = np.argmin(bics)
    best_k = ks[best_idx]
    best_result = results[best_idx]
    return best_k, best_result, liklihoods, bics

#!/usr/bin/env python3
"""K-means"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, liklie_hood = expectation(X, pi, m, S)
    for i in range(iterations):
        prev_l = liklie_hood
        if verbose and (i % 10) == 0:
            print('Log likelihood after {} iterations: {}'.format(
                i, liklie_hood.round(5)))
        pi, m, S = maximization(X, g)
        g, liklie_hood = expectation(X, pi, m, S)
        if abs(liklie_hood - prev_l) <= tol:
            break
    if verbose:
        print('Log likelihood after {} iterations: {}'.format(
            i + 1, liklie_hood.round(5)))
    return pi, m, S, g, liklie_hood

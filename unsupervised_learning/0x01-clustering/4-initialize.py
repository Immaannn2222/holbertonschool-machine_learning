#!/usr/bin/env python3
"""K-means"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initializes variables for a Gaussian Mixture Model"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    _, d = X.shape
    C, clss = kmeans(X, k)
    pi = 1 / k * np.ones(k)
    m = C
    S = np.array([np.identity(d)] * k)
    return pi, m, S

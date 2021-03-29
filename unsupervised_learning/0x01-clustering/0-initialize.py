#!/usr/bin/env python3
"""K-means"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    n, d = X.shape
    low_b = np.min(X, axis=0)
    high_b = np.max(X, axis=0)
    try:
        centroids = np.random.uniform(low_b, high_b, (k, d))
        return centroids
    except Exception:
        return None

#!/usr/bin/env python3
"""Multivariate_probab"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    cov = (np.matmul((X - mean).T, (X - mean))) / (n - 1)
    return mean, cov

#!/usr/bin/env python3
"""Multivariate_probab"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix"""
    d, d = C.shape
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    X = np.sqrt(np.diag(C))
    corr = 1 / (np.outer(X, X) / C)
    return corr

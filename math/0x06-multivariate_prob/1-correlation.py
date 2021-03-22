#!/usr/bin/env python3
"""Multivariate_probab"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix"""
    d, d = C.shape
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2:
        raise ValueError("C must be a 2D square matrix")
    X = np.diag(np.sqrt(np.diag(C)))
    Inv = np.linalg.inv(X)
    corr = np.matmul(np.matmul(Inv, C), Inv)
    return corr

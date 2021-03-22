#!/usr/bin/env python3
"""Multivariate_probab"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    X = np.diag(np.sqrt(np.diag(C)))
    Inv = np.linalg.inv(X)
    corr = np.matmul(np.matmul(Inv, C), Inv)
    return corr

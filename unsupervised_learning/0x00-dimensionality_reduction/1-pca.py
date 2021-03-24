#!/usr/bin/env python3
"""Dimensionality_reduction"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""
    X_mean = X - np.mean(X, axis=0)
    u, M, V = np.linalg.svd(X_mean)
    v = np.cumsum(M) / np.sum(M)
    W = V.T[:, :ndim]
    T = np.matmul(X_mean, W)
    return T

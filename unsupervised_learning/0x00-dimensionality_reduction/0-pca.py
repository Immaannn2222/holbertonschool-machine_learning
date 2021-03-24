#!/usr/bin/env python3
"""Dimensionality_reduction"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""
    U, T, V = np.linalg.svd(X)
    M = np.cumsum(T) / np.sum(T)
    i = np.where(M <= var, 1, 0)
    i = np.sum(i)
    return V.T[:, :i + 1]

#!/usr/bin/env python3
"""K-means"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set"""
    distance = np.linalg.norm(
        np.expand_dims(X, 2) - np.expand_dims(C.T, 0), axis=1)
    cluster = distance.argmin(axis=1)
    var = np.linalg.norm(X - C[cluster]) ** 2
    return var

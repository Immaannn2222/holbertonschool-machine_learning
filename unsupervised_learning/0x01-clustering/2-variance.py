#!/usr/bin/env python3
"""K-means"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set"""
    distance = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    clusters = distance.argmin(axis=0)
    var = np.linalg.norm(X - C[clusters]) ** 2
    return var

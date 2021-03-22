#!/usr/bin/env python3
"""Multivariate_probab"""
import numpy as np


class MultiNormal:
    """the class pf multinormal"""
    def __init__(self, data):
        """class constructor"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        Y = data - self.mean
        self.cov = (np.dot(Y, Y.T)) / (n - 1)

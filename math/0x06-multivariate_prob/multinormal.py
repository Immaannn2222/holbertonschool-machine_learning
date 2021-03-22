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

    def pdf(self, x):
        """calculates the PDF at a data point"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2 or x.shape[1] != 1\
           or x.shape[0] != self.mean.shape[0]:
            raise ValueError("x must have the shape ({}, 1)"
                             .format(self.mean.shape[0]))
        d = self.cov.shape[0]
        x_m = x - self.mean
        sqrt = np.sqrt((2 * np.pi)**d * np.linalg.det(self.cov))
        exp = np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2)
        return (1 / (sqrt) * exp)[0][0]

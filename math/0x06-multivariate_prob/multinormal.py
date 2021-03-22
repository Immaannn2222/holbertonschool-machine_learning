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
        d, _ = self.cov.shape
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        if x.ndim != 2 or (x.shape[0] != d)or (x.shape[1] != 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        me_an = x - self.mean
        st = np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        exp = np.exp(-(np.linalg.solve(self.cov, me_an).T.dot(me_an)) / 2)
        return (1 / (st) * exp)[0][0]

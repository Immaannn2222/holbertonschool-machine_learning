#!/usr/bin/env python3
"""HYPERPARAÙETER"""
import numpy as np


class GaussianProcess:
    """ represents a noiseless 1D Gaussian process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """class constructor"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix between two matrices"""
        exp_term = (X1 - X2.T) ** 2
        RBF = (((self.sigma_f) ** 2) * (np.exp(exp_term * (
            -0.5 / self.l ** 2))))
        return RBF

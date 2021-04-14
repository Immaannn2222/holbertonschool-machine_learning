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

    def predict(self, X_s):
        """predicts mean, standard deviation of points in a Gaussian process"""
        K_ss = self.kernel(X_s, X_s)
        K = self.kernel(self.X, self.X)
        decompositon = np.linalg.cholesky(K)
        K_k = self.kernel(self.X, X_s)
        result = np.linalg.solve(decompositon, K_k)
        mu = np.dot(result.T, np.linalg.solve(decompositon, self.Y)).reshape((
            X_s.shape[0],))
        s2 = np.diag(K_ss) - np.sum(result**2, axis=0)
        return mu, s2

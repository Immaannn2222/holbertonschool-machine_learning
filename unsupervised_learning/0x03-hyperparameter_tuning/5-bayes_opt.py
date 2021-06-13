#!/usr/bin/env python3
"""HYPERPARAÙETER"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """performs Bayesian optimization on a noiseless 1D Gaussian process"""
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """class constructor"""
        b_min, b_max = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.xsi = xsi
        self.X_s = np.linspace(b_min, b_max, ac_samples).reshape((-1, 1))
        self.minimize = minimize

    def acquisition(self):
        """calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is True:
            pred = np.min(self.gp.Y)
        else:
            pred = np.max(self.gp.Y)
        Z = (pred - mu - self.xsi) / sigma
        EI = (pred - mu - self.xsi) * norm.cdf(
            Z) + sigma * norm.pdf(Z)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """optimizes the black-box function"""

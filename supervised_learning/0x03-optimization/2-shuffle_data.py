#!/usr/bin/env python3
"""HYPERPARAMETER"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    M = np.random.permutation(X.shape[0])
    return X[M], Y[M]

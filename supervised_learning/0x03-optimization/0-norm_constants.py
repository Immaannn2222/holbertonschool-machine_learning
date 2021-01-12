#!/usr/bin/env python3
"""HYPERPARAMETER"""
import numpy as np


def normalization_constants(X):
    """ calculates the normalization (standardization) constants of a matrix"""
    me_an = np.mean(X, 0)
    return me_an, np.std(X, 0)

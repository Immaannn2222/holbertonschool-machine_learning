#!/usr/bin/env python3
"""HYPERPARAMETER"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output using batch normalization"""
    nu = np.mean(Z, 0)
    deno = np.var(Z, 0)
    Z_norm = (Z - nu) / np.sqrt(deno + epsilon)
    Z = gamma * Z_norm + beta
    return Z

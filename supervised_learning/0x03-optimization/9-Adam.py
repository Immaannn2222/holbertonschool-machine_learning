#!/usr/bin/env python3
"""HYPERPARAMETER"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable in place using the Adam optimization algorithm"""
    v = beta1 * v + (1 - beta1) * grad
    v_1 = v / (1-beta1**t)

    s = beta2*s + (1 - beta2) * grad ** 2
    s_1 = s / (1 - beta2 ** t)

    var = var - alpha * v_1 / (np.sqrt(s_1) + epsilon)
    return var, v, s

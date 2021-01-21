#!/usr/bin/env python3
"""Regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network with L2 regularization"""
    norm = np.linalg.norm
    s = []
    for i in range(L):
        s.append(norm(weights["W"+str(i + 1)]))

    L2 = cost + (lambtha / (2 * m)) * sum(s)
    return L2

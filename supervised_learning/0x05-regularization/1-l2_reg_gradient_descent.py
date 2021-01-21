#!/usr/bin/env python3
"""Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates gradient descent with L2 regularization"""
    l = len(Y[0])
    dZ = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        Act = cache["A" + str(i - 1)]
        dW = np.matmul(dZ, Act.T) / l
        db = np.sum(dZ, axis=1, keepdims=True) / l
        tanh = 1 - Act * Act
        dZ = np.matmul(weights[
            "W" + str(i)].T, dZ) * tanh
        dW_L2reg = dW + (lambtha/l)*weights["W" + str(i)]
        weights["W" + str(i)] -= alpha * dW_L2reg
        weights["b" + str(i)] -= alpha * db

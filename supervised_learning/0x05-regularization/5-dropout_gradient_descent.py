#!/usr/bin/env python3
"""Regularization"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights-Dropout regularization using gradient descent"""
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        tan_h = cache["A" + str(i - 1)]
        dW = np.matmul(dz, tan_h.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dtan_h = 1 - tan_h * tan_h
        dz = np.matmul(weights[
            "W" + str(i)].T, dz) * dtan_h
        if i > 1:
            dz = dz * cache["D" + str(i - 1)]
            dz = dz / keep_prob

        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db

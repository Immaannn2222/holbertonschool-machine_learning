#!/usr/bin/env python3
"""Regularization"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""
    c_dict = {"A0": X}
    for i in range(L):
        z = np.matmul(weights["W" + str(i + 1)],
                      c_dict["A" + str(i)]) + weights["b" + str(i + 1)]
        drop = np.random.binomial(1, keep_prob, size=z.shape)
        if i == L - 1:
            x = np.exp(z)
            c_dict["A" + str(i + 1)] = x / np.sum(x, axis=0, keepdims=True)
        else:
            c_dict["A" + str(i + 1)] = np.tanh(z)
            c_dict["D" + str(i + 1)] = drop
            c_dict["A" + str(i + 1)] = (c_dict["A" + str(i + 1)]
                                        ) * c_dict[
                                            "D" + str(i + 1)] / keep_prob
    return c_dict

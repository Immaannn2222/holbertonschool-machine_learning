#!/usr/bin/env python3
"""Convo Neural Network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    da_prev = np.zeros_like(A_prev)
    kh, kw = kernel_shape
    sh, sw = stride
    for i in range(m):
        for j in range(h_new):
            for x in range(w_new):
                for y in range(c_new):
                    part_A = A_prev[i, j * sh:j *
                                    sh + kh, x * sw:x * sw + kw, :]
                    if mode == "max":
                        res = (part_A == np.max(part_A))
                    else:
                        av = 1 / (kh * kw)
                        res = av * np.ones(kernel_shape)
                    da_prev[i, j * sh:j * sh + kh, x * sw:x *
                            sw + kw, :] += res * dA[i, j, x, y]
    return da_prev

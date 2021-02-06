#!/usr/bin/env python3
"""Convo Neural Network"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    o_h = int(((h_prev - kh) / sh) + 1)
    o_w = int(((w_prev - kw) / sw) + 1)

    output = np.zeros((m, o_h, o_w, c_prev))

    for i in range(o_h):
        for j in range(o_w):
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    A_prev[:, i * sh:i * sh + kh, j * sw: j * sw + kw, :],
                    axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(
                    A_prev[:, i * sh:i * sh + kh, j * sw: j * sw + kw, :],
                    axis=(1, 2))
    return output

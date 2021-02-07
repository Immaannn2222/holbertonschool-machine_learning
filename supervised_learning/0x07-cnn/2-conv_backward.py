#!/usr/bin/env python3
"""Convo Neural Network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """back propagation over a convolutional layer of a neural network"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        p_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        p_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        p_h = p_w = 0
    A_prev = np.pad(A_prev, ((0, 0), (p_h, p_h),
                             (p_w, p_w), (0, 0)), 'constant')
    dW = np.zeros_like(W)
    dA = np.zeros_like(A_prev)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for i in range(m):
        for j in range(h_new):
            for x in range(w_new):
                for y in range(c_new):
                    kernel = W[:, :, :, y]
                    dz = dZ[i, j, x, y]
                    part_A = A_prev[i, j * sh:j *
                                    sh + kh, x * sw:x * sw + kw, :]
                    dA[i,
                       j * sh:j * sh + kh,
                       x * sw:x * sw + kw,
                       :] += dz * kernel
                    dW[:, :, :, y] += part_A * dz
                    db[:, :, :, y] += dz
    dA = dA[:, p_h:dA.shape[1] - p_h, p_w:dA.shape[2] - p_w, :]
    return dA, dW, db

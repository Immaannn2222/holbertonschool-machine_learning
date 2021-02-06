#!/usr/bin/env python3
"""Convo Neural Network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """forward propagation over a convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        p_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        p_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        p_h = p_w = 0
    o_h = int(((h_prev - kh + 2 * p_h) / sh) + 1)
    o_w = int(((w_prev - kw + 2 * p_w) / sw) + 1)
    o_dim = (m, o_h, o_w, c_new)
    output = np.zeros(o_dim)
    pad_img = np.pad(A_prev, ((0, 0), (p_h, p_h),
                              (p_w, p_w), (0, 0)), 'constant')
    for i in range(o_h):
        for j in range(o_w):
            y = pad_img[:, i * sh:kh + i * sh, j * sw:kw + j * sw, :]
            for k in range(c_new):
                output[:, i, j, k] = (
                    y * W[:, :, :, k]
                ).sum(axis=(1, 2, 3))
    x = activation(output + b)
    return x

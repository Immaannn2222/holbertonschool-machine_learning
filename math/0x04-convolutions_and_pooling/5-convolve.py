#!/usr/bin/env python3
"""Convolve"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolution on images using multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride
    if padding == 'valid':
        p_h = 0
        p_w = 0

    elif padding == 'same':
        p_h = int((((h - 1) * sh + kh - h) / 2)) + 1
        p_w = int((((w - 1) * sw + kw - w) / 2)) + 1

        output = np.zeros((m, h, w))

    else:
        p_h = padding[0]
        p_w = padding[1]

    o_h = int(((h - kh + 2 * p_h) / sh) + 1)
    o_w = int(((w - kw + 2 * p_w) / sw) + 1)

    pad_img = np.zeros((m, h + o_h, w + o_w, c))
    pad_img = np.pad(images, ((0, 0), (p_h, p_h),
                              (p_w, p_w), (0, 0)), 'constant')

    output = np.zeros((m, o_h, o_w, nc))

    for x in range(nc):
        ke = (kernels[:, :, :, x])
        for i in range(o_h):
            for j in range(o_w):
                output[:, i, j, x] = (
                    pad_img[:, i * sh:kh + i * sh, j * sw:kw + j * sw, :] * ke
                ).sum(axis=(1, 2, 3))
    return output

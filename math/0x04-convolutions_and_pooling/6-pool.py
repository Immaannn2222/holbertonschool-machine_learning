#!/usr/bin/env python3
"""Convolve"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    o_h = int(((h - kh) / sh) + 1)
    o_w = int(((w - kw) / sw) + 1)

    output = np.zeros((m, o_h, o_w, c))

    for i in range(o_h):
        for j in range(o_w):
            if mode == 'max':
                output[:, x, y, :] = np.max(
                    images[:, i * sh:i * sh + kh, j * sw: j * sw + kw, :],
                    axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(
                    images[:, i * sh:i * sh + kh, j * sw: j * sw + kw, :],
                    axis=(1, 2))
    return output

#!/usr/bin/env python3
"""Convolve"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images"""
    m, h, w, c = images.shape
    sh, sw = stride

    o_h = int(((h - kernel_shape[0]) / sh) + 1)
    o_w = int(((w - kernel_shape[1]) / sw) + 1)

    output = np.zeros((m, o_h, o_w, c))

    for i in range(o_h):
        for j in range(o_w):
            if mode == 'max':
                output[:, x, y, :] = np.max(
                    images[:, i * sh:i * sh + kernel_shape[0], j * sw: j * sw + kernel_shape[1], :],
                    axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(
                    images[:, i * sh:i * sh + kernel_shape[0], j * sw: j * sw + kernel_shape[1], :],
                    axis=(1, 2))
        return output

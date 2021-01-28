#!/usr/bin/env python3
"""Convolve"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on images with channels"""
    m, h, w, d = images.shape
    kh, kw, kd = kernel.shape
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

    output = np.zeros((m, o_h, o_w))
    padded_img = np.pad(
        array=images,
        pad_width=((0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=0)

    for i in range(o_h):
        for j in range(o_w):
            output[:, i, j] = (
                padded_img[:, i * sh:kh + i * sh, j * sw:kw + j * sw] * kernel
                               ).sum(axis=(1, 2, 3))
    return output

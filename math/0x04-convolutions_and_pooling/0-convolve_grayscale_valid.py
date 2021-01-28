#!/usr/bin/env python3
"""Conv"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    o_h = h - kh + 1
    o_w = w - kw + 1
    output = np.zeros((m, o_h, o_w))
    for x in range(o_h):
        for y in range(o_w):
            output[:, x, y] = np.sum(
                kernel * images[:, x:x + kh, y:y + kw],
                axis=(1, 2)
            )
    return output
    return output

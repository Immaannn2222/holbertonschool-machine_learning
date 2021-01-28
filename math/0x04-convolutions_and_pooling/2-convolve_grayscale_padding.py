#!/usr/bin/env python3
"""Convolve"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    o_h = h + padding[0]
    o_w = w + padding[1] + padding[0]
    output = np.zeros((m, o_h, o_w))
    pad_images = np.pad(
        array=images,
        pad_width=((0,), (padding[0],), (padding[1],)),
        mode="constant")
    for x in range(o_h):
        for y in range(o_w):
            output[:, x, y] = np.sum(
                kernel * pad_images[:, x:x + kh, y:y + kw],
                axis=(1, 2)
            )
    return output

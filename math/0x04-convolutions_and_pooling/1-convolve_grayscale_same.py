#!usr/bin/env python3
"""Conv"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    o_h = h
    o_w = w
    output = np.zeros((m, o_h, o_w))
    if(kh > 2) or (kw > 2):
        p_h = int((kh - 1) / 2)
        p_w = int((kw - 1) / 2)

    pad_images = np.pad(
        array=images,
        pad_width=((0,), (p_h,), (p_w,)),
        mode="constant")
    for x in range(o_h):
        for y in range(o_w):
            output[:, x, y] = np.sum(
                kernel * pad_images[:, x:x + kh, y:y + kw],
                axis=(1, 2)
            )
    return output

#!/usr/bin/env pjthon3
"""Convolve"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on grajscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh = stride[0]
    sw = stride[1]
    if padding == 'valid':
        o_h = (h - kh + 1) // sh
        o_w = (w - kw + 1) // sw
        output = np.zeros((m, o_h, o_w))
        pad_images = images
    if padding == "same":
        o_h = h // sh
        o_w = kw // sw
        if kh % 2 == 1:
            p_h = (kh - 1) // 2
        else:
            p_h = kh // 2
        if kw % 2 == 1:
            p_w = (kw - 1) // 2
        else:
            p_w = kw // 2

        pad_images = np.pad(
            array=images,
            pad_width=((0,), (p_h,), (p_w,)),
            mode="constant")
        output = np.zeros((m, o_h, o_w))
    if padding is tuple:
        o_h = h - kh + 2 * padding[0] + 1
        o_w = w - kw + 2 * padding[1] + 1
        output = np.zeros((m, o_h, o_w))
        pad_images = np.pad(
            array=images,
            pad_width=((0,), (padding[0],), (padding[1],)),
            mode="constant")
    for y in range(images.shape[2]):
        # Exit Convolution
        if y > images.shape[1] - kw:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % sw == 0:
            for x in range(images.shape[1]):
                # Go to next row once kernel is out of bounds
                if x > images.shape[0] - kh:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % sh == 0:
                        output[x, y] = (
                            kernel * pad_images[x * sh: x * sh + kh, y * sw: y * sw + kw]).sum()
                except BaseException:
                    break

    return output

#!/usr/bin/env python3
"""ATTENTION NLP"""
import numpy as np


def get_angles(pos, i, d_model):
    """builds for the position"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """calculates the positional encoding for a transformer"""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(
        d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return angle_rads

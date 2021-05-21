#!/usr/bin/env python3
"""ATTENTION NLP"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding for a transformer"""
    position = np.arange(max_seq_len)
    freqs = (2 * (np.arange(dm) // 2) / dm)
    encode = position.reshape(-1, 1) * freqs.reshape(1, -1)
    encode[:, 1::2] = np.sin(encode[:, 1::2])
    encode[:, ::2] = np.cos(encode[:, ::2])
    return encode

#!/usr/bin/env python3
""" one_hot_decode"""
import numpy as np


def one_hot_decode(one_hot):
    """ converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray):
        return None
    return np.argmax(one_hot.T, axis=1)

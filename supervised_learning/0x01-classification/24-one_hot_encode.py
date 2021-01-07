#!/usr/bin/env python3
""" one_hot_encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    try:
        return np.eye(classes)[Y.reshape(-1)].T
    except Exception:
        return None

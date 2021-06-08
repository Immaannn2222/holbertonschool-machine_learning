#!/usr/bin/env python3
"""HIDDEN MARKOV MODEL"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing"""
    n, m = np.shape(P)
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if n != m:
        return None
    if np.any(P < 0):
        return None
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None
    return np.any(P.diagonal() == 1)

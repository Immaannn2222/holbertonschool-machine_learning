#!/usr/bin/env python3
"""BAYESIAN PROBAB"""
import numpy as np
from math import factorial as f


def likelihood(x, n, P):
    """calculates the likelihood of data various hypothetical probabilities"""
    if not isinstance(x, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an inte\
                         ger that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not ((0 <= P).all() and (P <= 1).all()):
        raise ValueError("All values in P must be in the range [0, 1]")
    combinaison = f(n) / (f(x) * f(n - x))
    R = combinaison * (P ** x) * (np.power((1 - P), (n - x)))
    return R

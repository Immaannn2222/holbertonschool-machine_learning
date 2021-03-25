#!/usr/bin/env python3
"""BAYESIAN PROBAB"""
import numpy as np
from math import factorial as f
from scipy import special


def likelihood(x, n, P):
    """calculates the likelihood of data various hypothetical probabilities"""
    combinaison = f(n) / (f(x) * f(n - x))
    R = combinaison * np.power(P, x) * (np.power((1 - P), (n - x)))
    return R


def intersection(x, n, P, Pr):
    """calculates the intersection with various hypothetical probabilities"""
    return likelihood(x, n, P) * Pr


def marginal(x, n, P, Pr):
    """calculates the marginal probability of obtaining the data"""
    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, P, Pr):
    """calculates the posterior probability"""
    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)


def posterior(x, n, p1, p2):
    """posterior probability within a specific range given the data"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is\
                         greater than or equal to 0")
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(p1, float) or 0 > p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or 0 > p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

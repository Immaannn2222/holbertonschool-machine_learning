#!/usr/bin/env python3
"""advanced linear algebra"""
import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    if not np.array_equal(matrix, matrix.T):
        return None

    if np.all(np.linalg.eigvals(matrix) > 0):
        return("Positive definite")
    elif np.all(np.linalg.eigvals(matrix) < 0):
        return("Negative definite")
    elif np.all(np.linalg.eigvals(matrix) >= 0):
        return("Positive semi-definite")
    elif np.all(np.linalg.eigvals(matrix) <= 0):
        return("Negative semi-definite")
    else:
        return("Indefinite")

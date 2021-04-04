#!/usr/bin/env python3
"""K-means"""
import numpy as np


def pdf(X, m, S):
    """probability density function of a Gaussian distribution"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[1] != S.shape[0]:
        return None
    _, d = X.shape
    if d != m.shape[0] or (d, d) != S.shape:
        return None
    A = (X - m).T
    var = (np.dot(np.dot(A.T, np.linalg.inv(S)), A)).diagonal()
    P_int = np.exp(-var / 2) / np.sqrt(
        np.abs((2 * np.pi) ** d * np.linalg.det(S)))
    PDF = np.where(P_int < 1e-300, 1e-300, P_int)
    return PDF

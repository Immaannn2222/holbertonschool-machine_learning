#!/usr/bin/env python3
"""HIDDEN MARKOV MODEL"""
import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain"""
    n, m = np.shape(P)
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if n != m:
        return None
    if (P == np.eye(n)).all():
        return None
    a, b = np.linalg.eig(P.T)
    res_l = []
    for i in range(len(a)):
        if np.allclose(a[i], 1):
            res_l.append(i)
    if len(res_l) == 1:
        res = (b[:, res_l[0]].T)/np.sum(np.abs(b[:, res_l[0]].T))
        return np.expand_dims((np.abs(res)), axis=0)
    else:
        return None

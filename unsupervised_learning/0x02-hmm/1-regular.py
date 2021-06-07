#!/usr/bin/env python3
"""HIDDEN MARKOV MODEL"""
import numpy as np


def regular(P):
    """dfghjkluytertyuio"""
    n, m = np.shape(P)
    if (P == np.eye(n)).all():
        return None
    a, b = np.linalg.eig(P.T)
    l = []
    for i in range(len(a)):
        if np.allclose(a[i], 1):
            l.append(i)
    if len(l) == 1:
        return np.abs(b[:, l[0]].T)/np.sum(np.abs(b[:, l[0]].T))
    else:
        return None

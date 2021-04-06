#!/usr/bin/env python3
"""HIDDEN MARKOV MODEL"""
import numpy as np


def markov_chain(P, s, t=1):
    """determines probability of a markov chain being in a particular state"""
    for i in range(t):
        s = np.dot(s, P)
    return s

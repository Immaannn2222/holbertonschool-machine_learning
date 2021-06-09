#!/usr/bin/env python3
"""HIDDEN MARKOV MODEL"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model"""
    try:
        T, = Observation.shape
        N, _ = Emission.shape
        X = np.zeros((N, T))
        X[:, T - 1] = np.ones(N)
        for t in range(T - 2, -1, -1):
            for j in range(N):
                X[j, t] = (X[:, t + 1] * Emission[:, Observation[
                    t + 1]]).dot(Transition[j, :])
        P = np.sum(X[:, 0] * Emission[:, Observation[0]] * Initial.T)
        return P, X
    except Exception:
        return None, None

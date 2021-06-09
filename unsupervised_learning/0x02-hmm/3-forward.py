#!/usr/bin/env python3
"""HIDDEN MARKOV MODEL"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model"""
    try:
        T, = Observation.shape
        N, _ = Emission.shape
        X = np.zeros((N, T))
        X[:, 0] = Initial.T * Emission[:, Observation[0]]
        for t in range(1, T):
            for j in range(N):
                X[j, t] = X[:, t - 1].dot(
                    Transition[:, j]) * Emission[j, Observation[t]]
        x = np.sum(X[:, T - 1])
        return x, X
    except Exception:
        return None, None

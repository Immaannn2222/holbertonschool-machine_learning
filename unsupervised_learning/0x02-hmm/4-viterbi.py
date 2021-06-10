#!/usr/bin/env python3
"""HIDDEN MARKOV MODEL"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """calculates the most likely sequence of hidden states for a hmm"""
    try:
        T, = Observation.shape
        N, M = Emission.shape
        X = np.zeros((N, T))
        Y = np.zeros((N, T))
        X[:, 0] = Initial.T * Emission[:, Observation[0]]
        for t in range(1, T):
            for j in range(N):
                probability = X[:, t - 1] * Emission[
                    j, Observation[t]] * Transition[:, j]
                X[j, t] = np.max(probability)
                Y[j, t] = np.argmax(probability, 0)
        P = np.max(X[:, T - 1])
        S = []
        last = np.argmax(X[:, T - 1])
        S.append(last)
        for i in range(T - 1, 0, -1):
            last = int(Y[last, i])
            S.append(last)
        return S[::-1], P
    except Exception:
        return None, None

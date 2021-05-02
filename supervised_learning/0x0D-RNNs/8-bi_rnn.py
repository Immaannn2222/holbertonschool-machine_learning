#!/usr/bin/env python3
"""RECURRENT NEURAL NETWORK"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """performs forward propagation for a bidirectional RNN"""
    T, m, i = X.shape
    o = bi_cell.by.shape[1]
    h = h_0.shape[1]
    f = np.zeros((T + 1, m, h))
    b = np.zeros((T + 1, m, h))
    Y = np.zeros((T, m, o))
    f[0] = h_0
    b[-1] = h_t
    for t in range(1, T + 1):
        f[t] = bi_cell.forward(f[t - 1], X[t - 1])
    for t in range(0, T)[::-1]:
        b[t] = bi_cell.backward(b[t + 1], X[t])

    H = np.concatenate((f[1:], b[:-1]), axis=-1)
    Y = bi_cell.output(H)
    return H, Y

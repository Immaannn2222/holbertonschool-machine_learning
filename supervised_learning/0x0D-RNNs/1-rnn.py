#!/usr/bin/env python3
"""RNN class"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN"""
    t, m, i = X.shape
    h, o = rnn_cell.Wy.shape
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    for timestep in range(1, t + 1):
        H[timestep], Y[timestep - 1] = rnn_cell.forward(
            H[timestep - 1], X[timestep - 1])
    return H, Y

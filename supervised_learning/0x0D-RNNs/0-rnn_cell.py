#!/usr/bin/env python3
"""RNN"""
import numpy as np
from scipy.special import softmax


class RNNCell:
    """first class"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step"""
        # Returns: h_next, y
        # h_next is the next hidden state
        # y is the output of the cell
        a = np.concatenate((h_prev, x_t), axis=1)
        h_t = np.tanh(np.matmul(a, self.Wh) + self.bh)
        y = np.dot(h_t, self.Wy) + self.by
        y = softmax(y, axis=1)
        return h_t, y

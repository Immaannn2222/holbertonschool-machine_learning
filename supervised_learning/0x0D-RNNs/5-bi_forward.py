#!/usr/bin/env python3
"""RECURRENT NEURAL NETWORK"""
import numpy as np


class BidirectionalCell:
    """that represents a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros(shape=(1, h))
        self.bhb = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """hidden state in the forward direction for one time step"""
        x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(x, self.Whf) + self.bhf)
        return h_next

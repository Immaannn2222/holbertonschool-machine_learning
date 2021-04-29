#!/usr/bin/env python3
"""RNN"""
import numpy as np


class RNNCell:
    """first class"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def softmax(self, v):
        """Calculates the softmax activation function"""
        prob = np.exp(v - np.max(v))
        return prob / prob.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step"""
        # Returns: h_next, y
        # h_next is the next hidden state
        # y is the output of the cell
        a = np.concatenate((h_prev, x_t), axis=1)
        h_t = np.tanh(np.matmul(a, self.Wh) + self.bh)
        y = np.dot(h_t, self.Wy) + self.by
        y = self.softmax(y)
        return h_t, y

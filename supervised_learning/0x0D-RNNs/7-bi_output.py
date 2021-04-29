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

    def backward(self, h_next, x_t):
        """hidden state in the backward direction for one time step"""
        x_back = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(x_back, self.Whb) + self.bhb)
        return h_prev

    def softmax(self, v):
        """Calculates the softmax activation function"""
        prob = np.exp(v - np.max(v))
        return prob / prob.sum(axis=1, keepdims=True)

    def output(self, H):
        """calculates all outputs for the RNN"""
        t = H.shape[0]
        y = []
        for i in range(t):
            y.append(self.softmax(np.dot(H[i], self.Wy) + self.by))
        y = np.array(y)
        return y

#!/usr/bin/env python3
"""RECUURRENT NEURAL NETWORK"""
import numpy as np


class GRUCell:
    """ represents a gated recurrent unit"""
    def __init__(self, i, h, o):
        """clas constructor"""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros(shape=(1, h))
        self.br = np.zeros(shape=(1, h))
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def softmax(self, v):
        """Calculates the softmax activation function"""
        prob = np.exp(v - np.max(v))
        return prob / prob.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Calculates the sigmoid"""
        return 1/(1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """function that performs forward propagation for one time step"""
        x = np.concatenate((h_prev, x_t), axis=1)
        s = self.sigmoid(np.dot(x, self.Wz) + self.bz)
        i = self.sigmoid(np.dot(x, self.Wr) + self.br)
        X = np.hstack(((i * h_prev), x_t))
        h = np.tanh(np.dot(X, self.Wh) + self.bh)
        h_next = s * h + (1 - s) * h_prev
        y = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y)
        return h_next, y

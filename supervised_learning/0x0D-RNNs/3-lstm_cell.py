#!/usr/bin/env python3
"""RECUURRENT NEURAL NETWORK"""
import numpy as np


class LSTMCell:
    """represents an LSTM unit"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros(shape=(1, h))
        self.bu = np.zeros(shape=(1, h))
        self.bc = np.zeros(shape=(1, h))
        self.bo = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def softmax(self, v):
        """Calculates the softmax activation function"""
        prob = np.exp(v - np.max(v))
        return prob / prob.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Calculates the sigmoid"""
        return 1/(1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """function that performs forward propagation for one time step"""
        x = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(np.dot(x, self.Wf) + self.bf)
        i = self.sigmoid(np.dot(x, self.Wu) + self.bu)
        C = np.tanh(np.matmul(x, self.Wc) + self.bc)
        C_t = f * c_prev + i * C
        o_o = self.sigmoid(np.matmul(x, self.Wo) + self.bo)
        h_next = o_o * np.tanh(C_t)
        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y)
        return h_next, C_t, y

#!/usr/bin/env python3
"""second class"""
import numpy as np


class NeuralNetwork:
    """the NeuralNetwork class"""
    def __init__(self, nx, nodes):
        """class instructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        "getter for weights"
        return self.__W1

    @property
    def b1(self):
        """bias getter"""
        return self.__b1

    @property
    def A1(self):
        """activated output getter"""
        return self.__A1

    @property
    def W2(self):
        "getter for weights"
        return self.__W2

    @property
    def b2(self):
        """bias getter"""
        return self.__b2

    @property
    def A2(self):
        """activated output getter"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        x = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + (np.exp(-x)))
        x2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + (np.exp(-x2)))
        return self.__A1, self.__A2

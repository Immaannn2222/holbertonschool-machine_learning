#!/usr/bin/env python3
""" third class """
import numpy as np


class Neuron:
    """ the neuron class"""
    def __init__(self, nx):
        """ class constructor """
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        "getter for weights"
        return self.__W

    @property
    def b(self):
        """bias getter"""
        return self.__b

    @property
    def A(self):
        """activated output getter"""
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation """
        m = np.shape(X)
        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + (np.exp(-x)))
        return self.__A

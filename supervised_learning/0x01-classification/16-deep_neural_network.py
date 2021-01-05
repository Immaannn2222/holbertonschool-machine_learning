#!/usr/bin/env python3
""" third class """
import numpy as np


class DeepNeuralNetwork:
    """ the neuron class"""

    def __init__(self, nx, layers):
        """class instructor"""
        self.nx = nx
        self.layers = layers
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not (all(layers) > 0) or not isinstance(all(layers), int):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for k in range(self.L):
            if k == 0:
                self.weights['W' + str(k + 1)
                             ] = np.random.randn(layers[0], nx) * np.sqrt(2/nx)
                self.weights['b' + str(k + 1)] = np.zeros((layers[0], 1))
            else:
                f = np.random.randn(layers[k], layers[k - 1]
                                    ) * np.sqrt(2/layers[k - 1])
                self.weights['W' + str(k + 1)] = f
                self.weights['b' + str(k + 1)] = np.zeros((layers[k], 1))

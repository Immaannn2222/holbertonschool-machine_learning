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
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if not (all(layers) >= 0):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers) - 1
        self.cache = {}
        self.weights = {}
        for l in range(0, self.L + 1):
            self.weights['W' + str(l)] = np.random.randn(layers[l], layers[l-1]) * np.sqrt(2/layers[l-1])
            self.weights['b' + str(l)] = np.zeros((layers[l], 1))

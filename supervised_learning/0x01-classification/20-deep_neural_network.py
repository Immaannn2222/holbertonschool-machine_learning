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
        if not all(
                map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for k in range(self.__L):
            if k == 0:
                self.__weights['W' + str(k + 1)] = np.random.randn(
                    layers[0], nx) * np.sqrt(2/nx)
                self.__weights['b' + str(k + 1)] = np.zeros((layers[0], 1))
            else:
                f = np.random.randn(layers[k], layers[k - 1]) * np.sqrt(
                    2/layers[k - 1])
                self.__weights['W' + str(k + 1)] = f
                self.__weights['b' + str(k + 1)] = np.zeros((layers[k], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache["A0"] = X
        for k in range(self.__L):
            d = np.matmul(self.__weights["W" + str(k + 1)], self.__cache[
                "A" + str(k)]) + (self.__weights["b" + str(k + 1)])
            r = self.__cache["A" + str(k + 1)] = 1/(1 + (np.exp(-d)))
        return r, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        i = np.shape(Y)[1]
        err_sum = 0.0
        err_sum = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        err_cost = -(1 / i) * err_sum
        return err_cost

    def evaluate(self, X, Y):
        """evaluates"""
        self.forward_prop(X)
        pred = np.where(self.__cache["A" + str(self.__L)] >= 0.5, 1, 0)
        return pred, self.cost(Y, self.__cache["A" + str(self.__L)])

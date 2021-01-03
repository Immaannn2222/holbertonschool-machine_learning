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
        x = np.dot(self.__W, X) + self.__b
        self.__A = 1/(1 + (np.exp(-x)))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression"""
        i = np.shape(Y)[1]
        print(i)
        err_sum = 0.0
        err_sum = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        err_cost = -(1 / i) * err_sum
        return err_cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        self.forward_prop(X)
        pred = np.where(self.A >= 0.5, 1, 0)
        return pred, self.cost(Y, self.A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron"""
        dW = np.dot((A - Y), X.T) / A.shape[1]
        self.__W = self.__W - alpha * dW
        db = np.sum((A - Y)) / A.shape[1]
        self.__b = self.__b - alpha * db
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for iteration in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)

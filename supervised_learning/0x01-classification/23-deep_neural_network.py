#!/usr/bin/env python3
""" third class """
import numpy as np
import matplotlib.pyplot as plt


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
        self.cache["A0"] = X
        for k in range(self.__L):
            d = np.matmul(self.__weights["W" + str(k + 1)], self.cache[
                "A" + str(k)]) + (self.__weights["b" + str(k + 1)])
            r = self.cache["A" + str(k + 1)] = 1/(1 + (np.exp(-d)))
        return r, self.cache

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
        pred = np.where(cache["A" + str(self.__L)] >= 0.5, 1, 0)
        return pred, self.cost(Y, self.cache["A" + str(self.__L)])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dz = self.cache["A" + str(self.L)] - Y
        for k in range(self.L, 0, -1):
            prev_lay = self.cache["A" + str(k - 1)]
            dW = np.matmul(dz, prev_lay.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            deriv_sig = prev_lay * (1 - prev_lay)
            dz = np.matmul(self.weights["W" + str(k)].T, dz) * deriv_sig
            self.__weights["W" + str(k)] = self.__weights[
                "W" + str(k)] - alpha * dW
            self.__weights["b" + str(k)] = self.__weights[
                "b" + str(k)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the deep neural network by updating the private attributes"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step not in range(0, iterations + 1):
                raise ValueError('step must be positive and <= iterations')
        i = []
        c = []
        for iteration in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if verbose and iteration % step == 0:
                c.append(self.cost(Y, self.__cache["A" + str(self.L)]))
                i.append(iteration)
                ind_cost = self.cost(Y, self.__cache["A" + str(self.L)])
                i.append(iteration)
                c.append(ind_cost)
                print("Cost after {} iterations: {}".format(
                    iteration, ind_cost))
        if graph:
            plt.plot(i, c)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

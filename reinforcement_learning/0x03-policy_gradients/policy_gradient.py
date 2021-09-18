#!/usr/bin/env python3
"""0x03-policy_gradients"""
import numpy as np


def softmax(X):
    """Computes the softmax"""
    i = np.exp(X - np.max(X))
    return i / np.sum(i)


def policy(matrix, weight):
    """ computes to policy with a weight of a matrix"""
    j = np.dot(matrix, weight)
    return softmax(j)


def softmax_grad(softmax):
    """Computes the gradient of a given softmax"""
    softmax = softmax.reshape(-1, 1)
    return np.diagflat(softmax) - softmax @ softmax.T


def policy_gradient(state, weight):
    """Computes the Monte Carlo weighted policy gradient."""
    x = policy(state, weight)
    action = np.random.choice(len(x[0]), p=x[0])
    y = softmax_grad(x)[action, :]
    s = y / x[0, action]
    g = np.dot(state.T, s[None, :])
    return action, g

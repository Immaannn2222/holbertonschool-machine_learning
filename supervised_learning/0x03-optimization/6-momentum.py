#!/usr/bin/env python3
"""HYPERPARAMETER"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """training operation using the gradient descent with momentum"""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)

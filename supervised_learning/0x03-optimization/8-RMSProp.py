#!/usr/bin/env python3
"""HYPERPARAMETER"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """training operation for a neural network """
    return tf.train.RMSPropOptimizer(
        alpha, decay=beta2, epsilon=epsilon).minimize(loss)

#!/usr/bin/env python3
"""Tensorflow project"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network"""
    tra_in = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return tra_in

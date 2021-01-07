#!/usr/bin/env python3
""" Tensorflow project """
import tensorflow as tf


def create_layer(prev, n, activation):
    """creates layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
            units=n,
            kernel_initializer=init,
            activation=activation, name="layer")
    return layer(prev)

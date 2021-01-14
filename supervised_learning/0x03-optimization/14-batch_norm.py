#!/usr/bin/env python3
"""HYPERPARAMETER"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """batch normalization layer for a neural network in tensorflow"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    prev_layer = tf.layers.dense(
        prev,
        n,
        activation=None,
        use_bias=True,
        kernel_initializer=initializer,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name='layer',
        reuse=None)
    m, v = tf.nn.moments(
        prev_layer, 0, shift=None, name=None
    )
    gamma = tf.zeros([n], tf.float32)
    beta = tf.ones([n], tf.float32)
    Z = tf.nn.batch_normalization(
        prev_layer,
        m,
        v,
        gamma,
        beta,
        1e-8,
        name=None
    )
    return activation(Z)

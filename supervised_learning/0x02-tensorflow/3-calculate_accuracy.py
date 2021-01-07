#!/usr/bin/env python3
"""Tensorflow project"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    pred = tf.math.argmax(y_pred, axis=1)
    eq = tf.math.equal(pred, tf.math.argmax(y, axis=1))
    accuracy = tf.math.reduce_mean(tf.cast(eq, tf.float32))
    return accuracy

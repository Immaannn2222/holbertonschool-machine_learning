#!/usr/bin/env python3
""" Tensorflow project """
import tensorflow as tf


def create_placeholders(nx, classes):
    """  returns two placeholders """
    x = tf.placeholder(tf.float32, name="x")
    y = tf.placeholder(tf.float32, name="y")
    return x, y

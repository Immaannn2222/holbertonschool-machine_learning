#!/usr/bin/env python3
""" Tensorflow project """
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ creates the forward propagation graph for the neural network"""
    lay_er = x
    for i in range(len(layer_sizes)):
        lay_er = create_layer(x, layer_sizes[i], activations[i])
    return lay_er

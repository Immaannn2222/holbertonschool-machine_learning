#!/usr/bin/env python3
"""Keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    input_1 = K.Input(shape=(nx,))
    layerr = input_1
    for i in range(len(layers)):
        if i != 0:
            layerr = K.layers.Dropout(lambtha)(layerr)
        layerr = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(layerr)
    return K.Model(inputs=input_1, outputs=layerr)

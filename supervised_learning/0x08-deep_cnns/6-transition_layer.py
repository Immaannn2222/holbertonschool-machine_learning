#!/usr/bin/env python3
"""deep CNN"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """transition layer in Densely Connected Convolutional Networks"""
    channel = int(nb_filters * compression)
    layer = K.layers.BatchNormalization()(X)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(
        channel,
        (1,
         1),
        kernel_initializer='he_normal',
        padding='same')(layer)
    layer = K.layers.AveragePooling2D((2, 2), strides=2)(layer)

    return layer, channel

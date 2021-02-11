#!/usr/bin/env python3
"""deep CNN"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """identity block in Deep Residual Learning for Image Recognition"""
    F11, F3, F12 = filters
    layer = A_prev
    layer_jump = layer
    layer = K.layers.Conv2D(filters=F11, kernel_size=(
        1, 1), kernel_initializer='he_normal')(layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(
        filters=F3,
        kernel_size=(
            3,
            3),
        padding='same',
        kernel_initializer='he_normal')(layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), strides=(
            1, 1), kernel_initializer='he_normal')(layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Add()([layer, layer_jump])
    layer = K.layers.Activation('relu')(layer)
    return layer

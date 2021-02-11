#!/usr/bin/env python3
"""deep CNN"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """builds a projection block as described in Deep Residual Learning"""
    F11, F3, F12 = filters
    layer = A_prev
    layer = K.layers.Conv2D(filters=F11, kernel_size=(
        1, 1), strides=s, kernel_initializer='he_normal')(layer)
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
            1, 1), kernel_initializer='he_normal')(layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    jump = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=s,
        kernel_initializer='he_normal',
        padding='same'
    )(A_prev)
    skip_connection = K.layers.BatchNormalization()(jump)

    layer = K.layers.Add()([layer, skip_connection])
    layer = K.layers.Activation('relu')(layer)
    return layer

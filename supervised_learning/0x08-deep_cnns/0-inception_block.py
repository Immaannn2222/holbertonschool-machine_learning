#!/usr/bin/env python3
"""deep CNN"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """inception block as described in Going Deeper with Convolutions (2014)"""
    F1, F3R, F3, F5R, F5, FPP = filters
    conv1x1 = K.layers.Conv2D(
        F1,
        kernel_size=(
            1,
            1),
        padding='same',
        activation="relu")(A_prev)
    conv_F3R = K.layers.Conv2D(
        F3R,
        kernel_size=(
            1,
            1),
        padding='same',
        activation="relu")(A_prev)
    conv3x3 = K.layers.Conv2D(
        F3,
        kernel_size=(
            3,
            3),
        padding='same',
        activation="relu")(conv_F3R)
    conv_F5R = K.layers.Conv2D(
        F5R,
        kernel_size=(
            1,
            1),
        padding='same',
        activation="relu")(A_prev)
    conv5x5 = K.layers.Conv2D(
        F5,
        kernel_size=(
            5,
            5),
        padding='same',
        activation="relu")(conv_F5R)
    pooling = K.layers.MaxPooling2D(
        (3, 3), strides=(
            1, 1), padding='same')(A_prev)
    conv_FPP = K.layers.Conv2D(
        FPP,
        kernel_size=(
            1,
            1),
        padding='same',
        activation="relu")(pooling)
    output_layer = K.layers.concatenate([conv1x1, conv3x3, conv5x5, conv_FPP])
    return output_layer

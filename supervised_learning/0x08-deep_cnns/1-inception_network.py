#!/usr/bin/env python3
"""deep CNN"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():

    X = K.Input(shape=(224, 224, 3))
    layer = K.layers.Conv2D(
        filters=64,
        kernel_size=(
            7,
            7),
        strides=2,
        padding='same',
        activation='relu')(X)
    layer = K.layers.MaxPooling2D(
        pool_size=(
            3,
            3),
        strides=2,
        padding='same')(layer)

    layer = K.layers.Conv2D(
        filters=64,
        kernel_size=(
            1,
            1),
        strides=1,
        padding='same',
        activation='relu')(layer)
    layer = K.layers.Conv2D(
        filters=192,
        kernel_size=(
            3,
            3),
        strides=1,
        padding='same',
        activation='relu')(layer)
    layer = K.layers.MaxPooling2D(
        pool_size=(
            3,
            3),
        strides=2,
        padding='same')(layer)

    layer = inception_block(layer, [64, 96, 128, 16, 32, 32])
    layer = inception_block(layer, [128, 128, 192, 32, 96, 64])
    layer = K.layers.MaxPooling2D(
        pool_size=(
            3,
            3),
        strides=2,
        padding='same')(layer)

    # stage-4
    layer = inception_block(layer, [192, 96, 208, 16, 48, 64])
    layer = inception_block(layer, [160, 112, 224, 24, 64, 64])
    layer = inception_block(layer, [128, 128, 256, 24, 64, 64])
    layer = inception_block(layer, [112, 144, 288, 32, 64, 64])
    layer = inception_block(layer, [256, 160, 320, 32, 128, 128])
    layer = K.layers.MaxPooling2D(
        pool_size=(
            3,
            3),
        strides=2,
        padding='same')(layer)

    layer = inception_block(layer, [256, 160, 320, 32, 128, 128])  # 5a
    layer = inception_block(layer, [384, 192, 384, 48, 128, 128])  # 5b
    layer = K.layers.AveragePooling2D(pool_size=(
        7, 7), strides=1, padding='valid')(layer)

    layer = K.layers.Dropout(0.4)(layer)

    output = K.layers.Dense(units=1000, activation='softmax')(layer)

    model = K.models.Model(inputs=X, outputs=output)

    return model

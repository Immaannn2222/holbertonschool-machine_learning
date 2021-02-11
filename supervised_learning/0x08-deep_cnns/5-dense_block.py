#!/usr/bin/env python3
"""deep CNN"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dense block as described"""
    for i in range(layers):
        batch_norm1 = K.layers.BatchNormalization(axis=3)(X)
        aciv1 = K.layers.Activation('relu')(batch_norm1)
        conv_1 = K.layers.Conv2D(filters=(4 * growth_rate),
                                 kernel_size=(1, 1),
                                 padding="same",
                                 kernel_initializer='he_normal')(aciv1)
        batch_norm2 = K.layers.BatchNormalization(axis=3)(conv_1)
        activ2 = K.layers.Activation('relu')(batch_norm2)
        conv_2 = K.layers.Conv2D(filters=growth_rate,
                                 kernel_size=(3, 3),
                                 padding="same",
                                 kernel_initializer='he_normal')(activ2)
        X = K.layers.concatenate([X, conv_2])
        nb_filters = nb_filters + growth_rate
    return X, nb_filters

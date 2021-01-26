#!/usr/bin/env python3
"""Keras"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        verbose=True,
        shuffle=False):
    """trains a model using mini-batch gradient descent"""
    history = network.fit(
        data,
        labels,
        shuffle=shuffle,
        epochs=epochs,
        verbose=verbose,
        batch_size=batch_size)
    return history

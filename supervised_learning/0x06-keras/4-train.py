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
        batch_size=batch_size,
        nb_epoch=epochs,
        verbose=verbose,
        shuffle=shuffle)
    return history

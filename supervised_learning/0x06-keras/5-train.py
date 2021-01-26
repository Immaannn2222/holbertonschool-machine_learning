#!/usr/bin/env python3
"""Keras"""
from tensorflow import keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        verbose=True,
        shuffle=False):
    """analyze validaiton data"""
    history = network.fit(
        x=data,
        y=labels,
        validation_data=(
            data,
            labels),
        shuffle=shuffle,
        nb_epoch=epochs,
        verbose=verbose,
        batch_size=batch_size)
    return history

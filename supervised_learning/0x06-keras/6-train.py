#!/usr/bin/env python3
"""Keras"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        early_stopping=False,
        patience=0,
        verbose=True,
        shuffle=False):
    """train the model using early stopping"""
    if early_stopping:
        ea_st = K.callbacks.EarlyStopping(patience=patience)
    history = network.fit(
        x=data,
        y=labels,
        validation_data=validation_data,
        shuffle=shuffle,
        nb_epoch=epochs,
        verbose=verbose,
        batch_size=batch_size,
        callbacks=[ea_st])
    return history

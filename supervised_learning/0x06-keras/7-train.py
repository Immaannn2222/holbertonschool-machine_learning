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
        learning_rate_decay=False,
        alpha=0.1,
        decay_rate=1,
        verbose=True,
        shuffle=False):
    """also train the model with learning rate decay"""
    if learning_rate_decay:
        def scheduler(step):
            """ scheduler Function """
            return alpha / (1 + decay_rate * step)
        l_dec = K.callbacks.LearningRateScheduler(
            scheduler, verbose=1)
        history = network.fit(x=data, y=labels, epochs=epochs,
                              verbose=verbose,
                              batch_size=batch_size,
                              validation_data=validation_data,
                              shuffle=shuffle, callbacks=[l_dec])
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
    else:
        history = network.fit(
            x=data,
            y=labels,
            validation_data=validation_data,
            shuffle=shuffle,
            nb_epoch=epochs,
            verbose=verbose,
            batch_size=batch_size)
    return history

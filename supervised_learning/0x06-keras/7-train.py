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
    callback_list = []
    if learning_rate_decay:
        def scheduler(step):
            return alpha / (1 + decay_rate * step)
        l_dec = K.callbacks.LearningRateScheduler(
            scheduler, verbose=1)
        callback_list.append(l_dec)
    if early_stopping:
        ea_st = K.callbacks.EarlyStopping(patience=patience)
        callback_list.append(ea_st)
        history = network.fit(
            x=data,
            y=labels,
            validation_data=validation_data,
            shuffle=shuffle,
            nb_epoch=epochs,
            verbose=verbose,
            batch_size=batch_size,
            callbacks=callback_list)
    else:
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

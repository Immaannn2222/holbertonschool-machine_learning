#!/usr/bin/env python3
from tensorflow import keras as K


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
    if early_stopping:
        ea_st = K.callbacks.EarlyStopping(patience=patience)
        history = network.fit(
            x=data,
            y=labels,
            validation_data=validation_data,
            shuffle=shuffle,
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            callbacks=[ea_st])
    if learning_rate_decay:
        def scheduler(epoch):
            return alpha / (1 + decay_rate) * epoch
        le_rate = K.callbacks.LearningRateScheduler(
            schedule=scheduler, verbose=1)
        history = network.fit(
            x=data,
            y=labels,
            validation_data=validation_data,
            shuffle=shuffle,
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            callbacks=[le_rate])
    return history

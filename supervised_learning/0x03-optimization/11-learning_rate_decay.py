#!/usr/bin/env python3
"""HYPERPARAMETER"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """learning rate decay"""
    return alpha / (1 + decay_rate * (global_step // decay_step))

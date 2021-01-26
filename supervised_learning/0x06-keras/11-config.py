#!/usr/bin/env python3
"""Keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves a model’s configuration in JSON format"""
    json_f = network.to_json()
    with open(filename, "w") as f:
        f.write(json_f)
    return None


def load_config(filename):
    """Loads a model withe a specefic configuration"""
    with open(filename, "r") as f:
        x = f.read()
    return K.models.model_from_json(x)

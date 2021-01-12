#!/usr/bin/env python3
"""HYPERPARAMETER"""
import numpy as np


def normalize(X, m, s):
    """normalizes (standardizes) a matrix"""
    return (X - np.mean(X, 0)) / np.std(X, 0)

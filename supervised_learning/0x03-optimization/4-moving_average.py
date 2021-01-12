#!/usr/bin/env python3
"""HYPERPARAMETER"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    
    bs = []
    b = 0
    for i in range(len(data)):
        b = beta * b + (1 - beta) * data[i]
        bc = b / (1 - beta ** (i + 1))
        bs.append(bc)
    return bs

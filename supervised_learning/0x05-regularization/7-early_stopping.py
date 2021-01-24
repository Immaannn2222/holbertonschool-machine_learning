#!/usr/bin/env python3
"""Regularization"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """function that determines if you should stop gradient descent early"""
    if opt_cost <= cost+threshold:
        count += 1
    else:
        count = 0
    return (patience == count), count

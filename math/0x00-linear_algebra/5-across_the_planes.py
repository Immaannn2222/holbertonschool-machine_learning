#!/usr/bin/env python3
""" first"""


def add_matrices2D(mat1, mat2):
    """add two array element wise"""
    sum_l = []
    for (x, y) in zip(mat1, mat2):
        sum_l.append(x + y)
    return sum_l

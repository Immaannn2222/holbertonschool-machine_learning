#!/usr/bin/env python3
""" first"""


def add_matrices2D(mat1, mat2):
    """add two array element wise"""
    sum_l = []
    if len(mat1) is not len(mat2) or len(mat1[0]) is not len(mat2[0]):
        return None
    for i in range(len(mat1)):
        for x, y in zip(mat1[i], mat2[i]):
            sum_l.append(x + y)
    return sum_l

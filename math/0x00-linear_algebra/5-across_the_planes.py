#!/usr/bin/env python3
""" first """


def add_matrices2D(mat1, mat2):
    """add two array element wise"""
    sum_l = []
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    sum_l = [[mat1[x][y] + mat2[x][y] for y in range(len(mat1[0]))]
             for x in range(len(mat1))]
    return sum_l

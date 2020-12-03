#!/usr/bin/env python3
"""returns the transpose of a 2D matrix, matrix"""


def matrix_transpose(matrix):
    """ the function"""
    for x in matrix:
        res = [[matrix[y][n] for y in range(len(matrix))]
               for n in range(len(matrix[0]))]
    return res

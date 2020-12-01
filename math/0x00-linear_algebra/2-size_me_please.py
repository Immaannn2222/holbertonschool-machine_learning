#!/usr/bin/env python3
"""calculates the shape of a matrix"""


def matrix_shape(matrix):
    """main fun"""
    if isinstance(matrix[0], list):
        shape = []
        new_matrix = append(matrix[0])
            shape = append(len(matrix[0]))
            matrix_shape(matrix[0])
    return shape

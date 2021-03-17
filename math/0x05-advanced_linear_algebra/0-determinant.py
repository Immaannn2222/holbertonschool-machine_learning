#!/usr/bin/env python3
"""advanced linear algebra"""


def determinant(matrix):
    """calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(i) != len(matrix) for i in matrix):
        raise ValueError('matrix must be a square matrix')
    if matrix == [[]]:
        return 1

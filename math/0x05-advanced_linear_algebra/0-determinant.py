#!/usr/bin/env python3
"""advanced linear algebra"""


def determinant(matrix):
    """function that calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or matrix[0] == []:
        raise TypeError('matrix must be a list of lists')
    if len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a square matrix')
    if matrix == [[]] or len(matrix[0]) == 0:
        return 1
    if len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    n = len(matrix)
    for i in range(n):
        x = [[matrix[i][j] for j in range(n)] for i in range(n)]
        x.pop(0)
        for m in range(n - 1):
            x[m].pop(i)
        det += ((-1)**i) * matrix[0][i] * determinant(x)
    return det

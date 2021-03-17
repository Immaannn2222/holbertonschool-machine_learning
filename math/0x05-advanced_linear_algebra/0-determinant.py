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
    lenth_m = len(matrix)
    matrix_cpy = matrix.copy()
    for d in range(lenth_m):
        for i in range(d + 1, lenth_m):
            if matrix_cpy[d][d] == 0:
                matrix_cpy[d][d] == 1e-18
            moving_s = matrix_cpy[i][d] / matrix_cpy[d][d]
            for j in range(lenth_m):
                matrix_cpy[i][j] = matrix_cpy[i][j] - movi\
                    ng_s * matrix_cpy[d][j]
    det = 1
    for i in range(lenth_m):
        det *= matrix_cpy[i][i]
    return det

#!/usr/bin/env python3
"""advanced linear algebra"""


def determinant(matrix):
    """calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        return 1
    if any(len(i) != len(matrix) for i in matrix):
        raise ValueError('matrix must be a square matrix')
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    # calculate the determinant of 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i in range(n):
        det += (-1) ** i * matrix[0][i] * determinant(reduce_mat(matrix, i))
    return det


def reduce_mat(matrix, x):
    """returns reduced matrix starting after the first linz"""
    # reducing the matrix/ elimminating column
    return [col[:x] + col[x + 1:] for col in (matrix[1:])]

# def reduce_mat_double(matrix, x, y):
#     """returns reduced matrix"""
#     # reducing the matrix/ elimminating column
#     return [col[:x] + col[x + 1:] for col in (matrix[:y] + matrix[y + 1:])]


def minor(matrix):
    """calculates the minor matrix of a matrix"""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(i) != len(matrix) for i in matrix):
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')
    n = len(matrix)
    if n == 1:
        return [[1]]
    if n == 2:
        return [elem[::-1] for elem in matrix][::-1]
    else:
        final_list = []
        for i in range(n):
            listt = []
            for j in range(n):
                new_mat = [col[:j] + col[j + 1:] for
                           col in (matrix[:i] + matrix[i + 1:])]
                listt.append(determinant(new_mat))
            final_list.append((listt))
        return final_list

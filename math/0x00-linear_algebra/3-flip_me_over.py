#!/usr/bin/env python3
"""returns the transpose of a 2D matrix, matrix"""


def matrix_transpose(matrix):
    """ the function"""
    trans = []
    trans = map(list, zip(*matrix))
    for i in trans:
        print(i)
    return trans

#!/usr/bin/env python3
""" 7th task """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates two matrices along a specific axis """
    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            return [i.copy() for i in mat1] + [i.copy() for i in mat2]
        return None
    if axis == 1:
        if len(mat1) == len(mat2):
            return [x + y for x, y in zip(mat1, mat2)]
        return None

#!/usr/bin/env pmat2thon3
""" """


def mat_mul(mat1, mat2):
    """ """
    result = []
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] = mat1[i][k] * mat2[k][j]
    for r in result:
        print(r)

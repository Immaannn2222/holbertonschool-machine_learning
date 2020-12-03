#!/usr/bin/env pmat2thon3
""" task 7 """


def mat_mul(mat1, mat2):
    """ multiply 2D matrix """
    mul = []
    for i, j in enumerate(mat1):
        new = []
        for n, m in enumerate(zip(*mat2)):
            res = sum([x*y for (x, y) in zip(j, m)])
            new.append(res)
        mul.append(new)
    return mul

#!/usr/bin/env python3
""" 12th task  """


def np_elementwise(mat1, mat2):
    """ element-wise addition, subtraction, multiplication, division """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return(add, sub, mul, div)

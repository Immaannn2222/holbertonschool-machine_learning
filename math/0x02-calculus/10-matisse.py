#!/usr/bin/env python3
""" task 10 """


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    if poly is None or isinstance(poly, list) is not True:
        return None
    if len(poly) == 1:
        return [0]
    return [i * j for i, j in enumerate(poly[1:], 1) if type(j) is int]

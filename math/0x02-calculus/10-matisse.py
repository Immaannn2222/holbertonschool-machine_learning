#!/usr/bin/env python3
""" task 10 """


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    if poly is None:
        return None
    if len(poly) <= 1:
        return [0]
    if not isinstance(poly, list):
        return [i * j for i, j in enumerate(poly[1:], 1)]
    return None

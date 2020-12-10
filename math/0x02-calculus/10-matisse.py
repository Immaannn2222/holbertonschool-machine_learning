#!/usr/bin/env python3
""" task 10 """


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    if poly == [0] or len(poly) < 1:
        return [0]
    if not isinstance(enumerate(poly[1:], 0), (int, float)) or not isinstance(poly, list) or poly == []:
        return [i * j for i, j in enumerate(poly[1:], 1)]
    return None

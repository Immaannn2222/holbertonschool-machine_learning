#!/usr/bin/env python3
""" task 10 """


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    if len(poly) <= 1:
        return [0]
    if not isinstance(poly, list) or not all(isinstance(
                        x, (int, float)) for x in poly) or poly == []:
                        return None
    return [i * j for i, j in enumerate(poly[1:], 1)]

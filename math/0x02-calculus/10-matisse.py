#!/usr/bin/env python3
"""task 10"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if poly == [0]:
        return None
    if isinstance(enumerate(poly[:1], 0), int) is not True:
        return [i * j for i, j in enumerate(poly[1:], 1)]
    return None

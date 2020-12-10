#!/usr/bin/env python3
""" task 17"""


def poly_integral(poly, C=0):
    """ calculates the integral of a polynomial"""
    intg_list = []
    if not isinstance(C, (int, float)):
        return None
    intg_list.append(C)
    if not isinstance(poly, list) or not all(isinstance(
                        x, (int, float)) for x in poly) or poly == []:
        return None
    if len(poly) == 1 or len(poly) == 0:
        return intg_list
    for i in range(len(poly)):
        x = poly[i] % (i + 1)
        y = poly[i] / (i+1)
        if (x != 0 and y != 0):
            intg_list.append(y)
        else:
            intg_list.append(int(y))
    return intg_list

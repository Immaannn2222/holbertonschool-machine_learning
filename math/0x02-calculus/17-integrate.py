#!/usr/bin/env python3
""" task 17"""


def poly_integral(poly, C=0):
    """ calculates the integral of a polynomial"""
    intg_list = []
    if not isinstance(poly, list) or poly == []:
        return None
    if not isinstance(C, (int, float)):
        return None
    intg_list.append(C)
    for i in range(len(poly)):
        intg_list.append(poly[i] / (i + 1))
    return intg_list

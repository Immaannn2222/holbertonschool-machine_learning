#!/usr/bin/env python3
""" 9th task """


def summation_i_squared(n): 
    """ calculates sum of i^2 """
    if not isinstance (n, (int, float)) or n < 1:
        return None
    return  (n * (n + 1) * (2 * n + 1)) // 6

#!/usr/bin/env python3
"""9th task"""


def summation_i_squared(n): 
    """calculates sum of i^2"""
    if isinstance (n, int) is not True or n < 0:
        return None
    return  (n * (n + 1) * (2 * n + 1)) // 6

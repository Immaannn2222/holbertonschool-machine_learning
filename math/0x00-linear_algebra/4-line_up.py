#!/usr/bin/env python3
""" first"""


def add_arrays(arr1, arr2):
    """add two arrays"""
    sum_arr = []
    if len(arr1) is not len(arr2):
        return None
    for i in range(0, len(arr1)):
        sum_arr.append(arr1[i] + arr2[i])
    return sum_arr

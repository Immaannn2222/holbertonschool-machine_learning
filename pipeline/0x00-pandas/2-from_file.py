#!/usr/bin/env python3
"""pandas dataframe"""
import pandas as pd


def from_file(filename, delimiter):
    """loads data from a file as a pd.DataFrame"""
    return pd.read_csv(filename, sep=delimiter)

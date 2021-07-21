#!/usr/bin/env python3
"""pandas dataframe"""
import pandas as pd


def from_numpy(array):
    """creates a pd.DataFrame from a np.ndarray"""
    _, n = array.shape
    alphabet = list(map(chr, range(ord('A'), ord('Z')+1)))
    df = pd.DataFrame(array, columns=[alphabet[i] for i in range(n)])
    # df.sort_index(axis=1)
    return df

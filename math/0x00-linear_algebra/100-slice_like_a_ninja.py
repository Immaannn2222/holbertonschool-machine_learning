#!/usr/bin/env python3
""" advanced task """


def np_slice(matrix, axes={}):
    """ slices a matrix along specific axes """
    new = []
    for x in range(len(matrix.shape)):
        if x in axes:
            new.append(slice(*axes[x]))
        else:
            new.append(slice(None))
    return(matrix[tuple(new)])

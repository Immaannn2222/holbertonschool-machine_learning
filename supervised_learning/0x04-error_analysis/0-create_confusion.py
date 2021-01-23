#!/usr/bin/env python3
"""Error Analysis"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix"""
    m, clas = labels.shape
    confusion = np.zeros((clas, clas))
    for k in range(m):
        i = np.nonzero(labels[k, :])
        j = np.nonzero(logits[k, :])
        confusion[i, j] += 1
    return confusion

#!/usr/bin/env python3
"""Error Analysis"""
import numpy as np


def specificity(confusion):
    """"specificity for each class in a confusionusion matrix"""
    TP = confusion.diagonal()
    FP = confusion.sum(axis=0) - TP
    FN = confusion.sum(axis=1) - TP
    TN = confusion.sum() - (TP+FP+FN)
    return TN / (TN + FP)

#!/usr/bin/env python3
"""K-means"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    if not isinstance(k, int) or k <= 0:
        return None
    try:
        n, d = X.shape
        low_b = np.min(X, axis=0)
        high_b = np.max(X, axis=0)
        centroids = np.random.uniform(low_b, high_b, (k, d))
    except Exception:
        return None
    return centroids


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    centroids = initialize(X, k)
    centroids_copy = np.copy(centroids)
    n, d = X.shape
    if centroids is None:
        return None, None
    for j in range(iterations):
        distance = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
        clusters = distance.argmin(axis=0)
        for i in range(k):
            assigned = np.argwhere(clusters == i)
            if (len(assigned) == 0):
                centroids[i] = initialize(X, 1)
            else:
                centroids[i] = np.mean(X[assigned], axis=0)
        if (centroids == centroids_copy).all():
            break
    distance = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
    clusters = distance.argmin(axis=0)
    return centroids, clusters

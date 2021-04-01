#!/usr/bin/env python3
"""K-means"""
import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_

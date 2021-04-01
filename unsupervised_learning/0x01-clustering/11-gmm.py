#!/usr/bin/env python3
"""K-means"""
import sklearn.mixture


def gmm(X, k):
    """calculates a GMM from a dataset"""
    GMM = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = GMM.weights_
    m = GMM.means_
    S = GMM.covariances_
    clss = GMM.predict(X)
    bic = GMM.bic(X)
    return pi, m, S, clss, bic

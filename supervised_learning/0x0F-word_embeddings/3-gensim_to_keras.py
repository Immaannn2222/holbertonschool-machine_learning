#!/usr/bin/env python3
"""NLP MODULE"""
from keras.layers.embeddings import Embedding


def gensim_to_keras(model):
    """converts a gensim word2vec model to a keras Embedding layer"""
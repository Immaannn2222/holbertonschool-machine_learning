#!/usr/bin/env python3
"""NLP MODULE"""


def gensim_to_keras(model):
    """converts a gensim word2vec model to a keras Embedding layer"""
    emebed_layer = model.wv.get_keras_embedding(train_embeddings=False)
    return emebed_layer

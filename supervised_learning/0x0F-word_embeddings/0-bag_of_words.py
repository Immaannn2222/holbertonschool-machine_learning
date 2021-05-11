#!/usr/bin/env python3
"""NLP"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix"""
    Vec = CountVectorizer(vocabulary=vocab)
    tokens = Vec.fit_transform(sentences)
    feature = Vec.get_feature_names()
    embedding = tokens.toarray()
    return feature, embedding

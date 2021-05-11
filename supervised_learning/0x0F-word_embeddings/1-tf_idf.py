#!/usr/bin/env python3
"""NLP"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding"""
    Vec = CountVectorizer(vocabulary=vocab)
    tokens = Vec.fit_transform(sentences)
    feature = Vec.get_feature_names()
    tfidf = TfidfTransformer()
    embeddings = tfidf.fit_transform(tokens).toarray()
    return embeddings, feature

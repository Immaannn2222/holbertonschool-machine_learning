#!/usr/bin/env python3
"""NLP MODULE"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """creates and trains a genism fastText model"""
    return FastText(
        sentences,
        size=size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=not cbow,
        iter=iterations,
        seed=seed,
        workers=workers
    )

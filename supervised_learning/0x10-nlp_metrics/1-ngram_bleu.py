#!/usr/bin/env python3
"""NLP METRICS"""
import numpy as np


def NGrams_tokenize(wordlist, n):
    """tokenizes sentence into n tokens"""
    ngrams_sentence = []
    for i in range(len(wordlist) - n + 1):
        ngrams_sentence.append(' '.join(wordlist[i:i + n]))
    return ngrams_sentence


def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence"""
    unigrams = len(sentence)
    token = np.array([len(r) for r in references])
    idx = np.argmin(np.abs(token - unigrams))
    r = len(references[idx])
    if r > unigrams:
        bp = np.exp(1 - r / unigrams)
    else:
        return 1
    references = NGrams_tokenize(r, n)
    sentence = NGrams_tokenize(sentence, n)
    words = {}
    for word in sentence:
        for ref in references:
            if word in words:
                if words[word] < ref.count(word):
                    words.update({word: ref.count(word)})
            else:
                words.update({word: ref.count(word)})
    p = sum(words.values()) / len(word)
    return bp * p

#!/usr/bin/env python3
"""NLP METRICS"""
import numpy as np


def NGrams_tokenize(wordlist, n):
    """tokenizes sentence into n tokens"""
    ngrams_sentence = []
    for i in range(len(wordlist) - (n - 1)):
        ngrams_sentence.append(' '.join(wordlist[i: i + n]))
    return ngrams_sentence


def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence"""
    unigrams = len(sentence)
    token = np.array([len(x) for x in references])
    idx = np.argmin(np.abs(token - unigrams))
    x = len(references[idx])
    if x > unigrams:
        bp = np.exp(1 - x / unigrams)
    else:
        bp = 1
    references = list(NGrams_tokenize(x, n) for x in references)
    sentence = NGrams_tokenize(sentence, n)
    words = {}
    for word in sentence:
        for ref in references:
            if word in words:
                if words[word] < ref.count(word):
                    words.update({word: ref.count(word)})
            else:
                words.update({word: ref.count(word)})
    p = (sum(words.values())) / len(sentence)
    return bp * p


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence """
    sen_size = len(sentence)
    refs = np.array([len(x) for x in references])
    min_indx = np.argmin(np.abs(refs - sen_size))
    x = len(references[min_indx])
    if x > sen_size:
        bp = np.exp(1 - x / sen_size)
    else:
        bp = 1
    ngrams = []
    for i in range(1, n + 1):
        ngrams.append(ngram_bleu(references, sentence, i))
    ngrams = np.array(ngrams)
    score = np.exp(np.sum((1 / n) * np.log(ngrams)))
    return bp * score

#!/usr/bin/env python3
"""TRANSFORMER APP"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """loads and preps a dataset for machine translation"""
    def __init__(self, batch_size, max_len):
        """class constructor"""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', split=[
                'train', 'validation'], as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        tokenize = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        en = tokenize((
            en.numpy() for pt, en in data), target_vocab_size=2**15)
        pt = tokenize((
            pt.numpy() for pt, en in data), target_vocab_size=2**15)
        return pt, en

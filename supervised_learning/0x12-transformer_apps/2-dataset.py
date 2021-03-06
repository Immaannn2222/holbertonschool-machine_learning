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

    def encode(self, pt, en):
        """encodes a translation into tokens"""
        pt_size = self.tokenizer_pt.vocab_size
        en_size = self.tokenizer_en.vocab_size
        pt = [pt_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_size + 1]
        en = [en_size] + self.tokenizer_en.encode(
            en.numpy()) + [en_size + 1]
        return pt, en

    def tf_encode(self, pt, en):
        """acts as a tensorflow wrapper for the encode instance method"""
        pt_wrap, en_wrap = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        pt_wrap.set_shape([None])
        en_wrap.set_shape([None])
        return pt_wrap, en_wrap

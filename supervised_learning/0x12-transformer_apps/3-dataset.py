#!/usr/bin/env python3
"""TRANSFORMER APP"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """loads and preps a dataset for machine translation"""
    def __init__(self, batch_size, max_len):
        """class constructor"""
        (data_train, data_valid), infos = tfds.load(
            'ted_hrlr_translate/pt_to_en', split=[
                'train', 'validation'
                ], as_supervised=True, shuffle_files=True, with_info=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train)
        buffer_size = infos.splits['train'].num_examples
        data_train = data_train.map(self.tf_encode)

        def filter_max_length(x, y, max_length=max_len):
            """ filtering out sentences with length > max_length"""
            return tf.logical_and(
                tf.size(x) <= max_length, tf.size(y) <= max_length)

        data_train = data_train.filter(filter_max_length)
        data_train = data_train.cache()
        data_train = data_train.shuffle(buffer_size).padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

        data_valid = data_valid.map(self.tf_encode)
        data_valid = data_valid.filter(filter_max_length)
        self.data_valid = data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset"""
        tokenize = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        en = tokenize((en.numpy() for pt, en in data), target_vocab_size=2**15)
        pt = tokenize((pt.numpy() for pt, en in data), target_vocab_size=2**15)
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

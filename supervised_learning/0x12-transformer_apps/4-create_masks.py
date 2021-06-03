#!/usr/bin/env python3
"""TRANSFORMER APP"""


def create_masks(inputs, target):
    """creates all masks for training/validation"""
    batch_size, seq_len_out = target.shape
    batch_size, seq_len_in = inputs.shape

    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)[
        :, tf.newaxis, tf.newaxis, :]
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)[
        :, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = tf.math.subtract(1, tf.linalg.band_part(
        tf.ones((batch_size, 1, seq_len_out, seq_len_out)), -1, 0))
    decoder_target_padding_mask = tf.cast(
        tf.math.equal(target, 0),
        tf.float32)[
        :, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)
    return encoder_mask, combined_mask, decoder_mask

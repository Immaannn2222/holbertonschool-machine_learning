#!/usr/bin/env python3
"""DATA AUGMENTATION"""
import tensorflow as tf


def crop_image(image, size):
    """performs a random crop of an image"""
    return tf.image.random_crop(image, size)

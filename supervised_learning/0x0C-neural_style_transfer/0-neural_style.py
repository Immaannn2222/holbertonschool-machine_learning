#!/usr/bin/env python3
"""TRANSFER LEARNING STYLE"""
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()



class NST():
    """performs tasks for neural style transfer"""
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """the class constructor"""
        if not isinstance(
            style_image,
            np.ndarray) and \
                style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(
            content_image,
            np.ndarray) and \
                content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (float, int)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """preprocess the image"""
        if not isinstance(
                image,
                np.ndarray) and image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        img_h, img_w, _ = image.shape
        img_max_dim = 512
        x = max(img_h, img_w)
        scale = img_max_dim / x
        n_shape = (int(img_h * scale), int(img_w * scale))
        image = np.expand_dims(image, axis=0)
        resize_img = tf.image.resize_bicubic(image, n_shape)
        scaled_img = tf.clip_by_value(resize_img / 255, 0, 1)
        return scaled_img

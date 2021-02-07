#!/usr/bin/env python3
"""Convo Networks"""
import tensorflow as tf


def lenet5(x, y):
    """builds a modified version of the LeNet-5 architecture"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_IN")
    cv1 = tf.nn.relu(tf.layers.Conv2D(6, (5, 5),
                                      padding='same',
                                      kernel_initializer=initializer,
                                      )(x))
    max_pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv1)
    cv2 = tf.nn.relu(tf.layers.Conv2D(16, (5, 5),
                                      padding='valid',
                                      kernel_initializer=initializer,
                                      )(max_pool))
    max_pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2)
    FC = tf.layers.Flatten()(max_pool2)
    FC1 = tf.nn.relu(tf.layers.Dense(120, kernel_initializer=initializer)(FC))
    FC2 = tf.layers.Dense(84, kernel_initializer=initializer,
                          activation='relu')(FC1)
    y_ = tf.layers.Dense(10, kernel_initializer=initializer)(FC2)
    y_pred = tf.nn.softmax(y_)
    loss = tf.losses.softmax_cross_entropy(y, y_)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(
        tf.math.argmax(y, axis=1),
        tf.math.argmax(y_, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss, acc

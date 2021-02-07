#!/usr/bin/env python3
"""Convo Neural Networks"""
import tensorflow.keras as K


def lenet5(X):
    """builds a modified version of the LeNet-5 architecture"""
    initializer = K.initializers.he_normal()
    cv1 = K.layers.Conv2D(6, (5, 5),
                          padding='same', kernel_initializer=initializer,
                          activation='relu')(X)
    max_pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv1)
    conv2 = K.layers.Conv2D(16, (5, 5),
                            padding='valid', kernel_initializer=initializer,
                            activation='relu')(max_pool1)
    max_pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    x = K.layers.Flatten()(max_pool2)
    y = K.layers.Dense(120, kernel_initializer=initializer,
                       activation='relu')(x)
    z = K.layers.Dense(84, kernel_initializer=initializer,
                       activation='relu')(y)
    output = K.layers.Dense(10, kernel_initializer=initializer,
                            activation='softmax')(z)
    model = K.models.Model(inputs=X, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

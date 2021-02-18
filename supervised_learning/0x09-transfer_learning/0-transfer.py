#!/usr/bin/env python3
"""TRANSFER LEARNING"""
from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """preprocess data"""
# the preprocess_input delivered from keras applications to preapre raw data
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)  # hot encode
    return X_p, Y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_val, Y_val = preprocess_data(X_val, Y_val)
    init = K.initializers.he_normal()  # initialize with he et al. method

    densenet_model = K.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling=None
    )
    # allows to train the last layers of the architecture where we get full
    # features
    densenet_model.trainable = True

    for output in densenet_model.layers:
        if 'conv5' in output.name:
            output.trainable = True
        else:
            output.trainable = False
# start adding layers
# the input my new model based on desnet
    input = K.Input(shape=(32, 32, 3))
    prepare = K.layers.Lambda(lambda x: tf.image.resize_images(x, (224, 224)),
                              name='lamb')(
        input)  # the resize_images doesn't work on colab
    output = densenet_model(inputs=prepare)
    output = K.layers.Flatten()(output)
    output = K.layers.BatchNormalization()(output)
    output = K.layers.Dense(units=256,
                            activation='relu',
                            kernel_initializer=init
                            )(output)
    output = K.layers.Dropout(0.4)(output)
    output = K.layers.BatchNormalization()(output)
    output = K.layers.Dense(units=128,
                            activation='relu',
                            kernel_initializer=init
                            )(output)
    output = K.layers.Dropout(0.4)(output)
    output = K.layers.Dense(units=10,
                            activation='softmax',
                            kernel_initializer=init
                            )(output)

    model = K.models.Model(inputs=input, outputs=output)

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    fitting = model.fit(
        X_train,
        Y_train,
        epochs=13,
        validation_data=(
            X_val,
            Y_val),
        batch_size=32,
        verbose=1)

    model.save('cifar10.h5')

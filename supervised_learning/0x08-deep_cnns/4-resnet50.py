#!/usr/bin/env python3
"""deep CNN"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ResNet-50 architecture"""
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                            padding='same',
                            strides=2,
                            kernel_initializer="he_normal")(X)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(batch1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same")(act1)
    proj0 = projection_block(pool1, [64, 64, 256], 1)
    iden1 = identity_block(proj0, [64, 64, 256])
    iden2 = identity_block(iden1, [64, 64, 256])
    proj2 = projection_block(iden2, [128, 128, 512])
    iden3 = identity_block(proj2, [128, 128, 512])
    iden4 = identity_block(iden3, [128, 128, 512])
    iden5 = identity_block(iden4, [128, 128, 512])
    proj2 = projection_block(iden5, [256, 256, 1024])
    id6 = identity_block(proj2, [256, 256, 1024])
    iden7 = identity_block(id6, [256, 256, 1024])
    iden8 = identity_block(iden7, [256, 256, 1024])
    ident9 = identity_block(iden8, [256, 256, 1024])
    iden10 = identity_block(ident9, [256, 256, 1024])
    p3 = projection_block(iden10, [512, 512, 2048])
    i_11 = identity_block(p3, [512, 512, 2048])
    i_12 = identity_block(i_11, [512, 512, 2048])
    average_pooling = K.layers.AveragePooling2D(pool_size=(7, 7),
                                                strides=(1, 1))(i_12)
    output = K.layers.Dense(units=1000, activation='softmax',
                            kernel_initializer="he_normal")(average_pooling)
    model = K.models.Model(inputs=X, outputs=output)
    return model

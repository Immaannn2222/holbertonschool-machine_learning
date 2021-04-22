#!/usr/bin/env python3
"""ENCODERS"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""
    UpSample = keras.layers.UpSampling2D
    Conv = keras.layers.Conv2D
    Max_p = keras.layers.MaxPooling2D

    model_input = keras.Input(input_dims)
    encoded = model_input
    for i in filters:
        encoded = Conv(i, (3, 3), activation='relu', padding='same')(encoded)
        encoded = Max_p((2, 2), padding='same')(encoded)
        encoder = keras.Model(model_input, encoded)

    latent_input = keras.Input(latent_dims)
    decoded = latent_input
    for i in (range(1, len(filters)))[::-1]:
        decoded = Conv(
            filters[i], (3, 3), activation='relu', padding='same')(decoded)
        decoded = UpSample((2, 2))(decoded)
    decoded = Conv(
        filters[0], (3, 3), activation='relu', padding='valid')(decoded)
    decoded = UpSample((2, 2))(decoded)
    decoded = Conv(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same')(decoded)
    decoder = keras.Model(latent_input, decoded)

    out_decoder = decoder(encoder(model_input))
    auto = keras.Model(model_input, out_decoder)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto

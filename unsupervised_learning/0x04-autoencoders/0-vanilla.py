#!/usr/bin/env python3
"""ENCODER"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates an auto"""
    Dense = keras.layers.Dense
    model_input = keras.Input((input_dims,))
    for hidden_layer in hidden_layers:
        encoded = Dense(hidden_layer, activation='relu')(model_input)
    encoded = Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(model_input, encoded)
    decode_0 = keras.Input((latent_dims,))
    decoded = decode_0
    for hidden_layer in (hidden_layers)[::-1]:
        decoded = Dense(hidden_layer, activation='relu')(decoded)
    decoded = Dense(input_dims, activation="sigmoid")(decoded)
    decoder = keras.Model(decode_0, decoded)
    out_decoder = decoder(encoder(model_input))
    auto = keras.Model(model_input, out_decoder)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto

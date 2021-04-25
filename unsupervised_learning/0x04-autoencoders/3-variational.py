#!/usr/bin/env python3
"""ENCODERS"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""
    Dense = keras.layers.Dense
    Lambda = keras.layers.Lambda

    model_input = keras.Input(input_dims)
    encoded = model_input
    for hidden_layer in hidden_layers:
        x = Dense(hidden_layer, activation="relu")(model_input)
    z_mean = Dense(latent_dims, activation=None)(x)
    z_log_sigma = Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Create a sampling layer"""
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon
    l_layer = Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(model_input, [z_mean, z_log_sigma, l_layer])
    latent = keras.Input(latent_dims)
    for hidden_layer in reversed(hidden_layers):
        y = Dense(hidden_layer, activation='relu')(latent)
    outputs = Dense(input_dims, activation='sigmoid')(y)
    decoder = keras.Model(latent, outputs)
    outputs = decoder(encoder(model_input))
    V = keras.Model(model_input, outputs)

    def VAE_loss(y, pred):
        V_loss = keras.losses.binary_crossentropy(model_input, outputs)
        V_loss *= input_dims
        n = 1 + z_log_sigma - keras.backend.square(
            z_mean) - keras.backend.exp(z_log_sigma)
        n = keras.backend.sum(n, axis=-1)
        n *= -0.5
        vae_loss = keras.backend.mean(V_loss + n)
        return vae_loss
    V.compile(optimizer="adam", loss=VAE_loss)
    return encoder, decoder, V

#!/usr/bin/env python3
"""ENCODERS"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""
    K = keras.backend
    model_input = keras.Input(shape=(input_dims,))
    hidden_l = keras.layers.Dense(units=hidden_layers[0], activation='relu')
    ltent_input = hidden_l(model_input)
    for hidden_layer in hidden_layers:
        h = keras.layers.Dense(
            hidden_layer, activation='relu')
        ltent_input = hidden_l(ltent_input)
    latent = keras.layers.Dense(units=latent_dims, activation=None)
    mean = latent(ltent_input)
    sigma = latent(ltent_input)

    def sampling(args):
        mean, sigma = args
        eps = keras.backend.random_normal(shape=(latent_dims,), mean=0.0,
                                          stddev=1.0)
        return mean + keras.backend.exp(sigma) * eps

    lambda_layer = keras.layers.Lambda(sampling, output_shape=(
        latent_dims,))([mean, sigma])
    encoder = keras.models.Model(model_input, lambda_layer)
    latent_input = keras.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers) - 1, -1, -1):
        if i == len(hidden_layers) - 1:
            decoder = keras.layers.Dense(
                hidden_layers[i], activation='relu')(latent_input)
        else:
            decoder = keras.layers.Dense(
                hidden_layers[i], activation='relu')(decoder)
    decoder = keras.layers.Dense(input_dims, activation='sigmoid')(decoder)
    M_decode = keras.models.Model(latent_input, decoder)
    ltent_input = keras.Input(shape=(input_dims,))
    s = M_decode(encoder(ltent_input))
    autoenc = keras.models.Model(inputs=ltent_input, outputs=s)

    def vae_loss(ltent_input, s):
        loss_bn = keras.backend.binary_crossentropy(ltent_input, s)
        loss_back = -0.5 * keras.backend.mean(
            1 + sigma - K.square(mean) - K.exp(
                sigma), axis=-1)
        return loss_bn + loss_back
    autoenc.compile(optimizer='Adam', loss=vae_loss)
    return encoder, M_decode, autoenc

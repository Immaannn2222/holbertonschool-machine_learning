#!/usr/bin/env python3
"""ATTENTION NLP"""
import tensorflow


class RNNEncoder(tensorflow.keras.layers.Layer):
    """encode for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        """class constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tensorflow.keras.layers.Embedding(vocab, embedding)
        self.gru = tensorflow.keras.layers.GRU(
            units=units, kernel_initializer='glorot_unif\
                orm', return_sequences=True, return_state=True)

    def initialize_hidden_state(self):
        """Initializes the hidden states to a tensor of zeros"""
        return tensorflow.zeros((self.batch, self.units))

    def call(self, x, initial):
        """call function"""
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded, initial)
        return outputs, hidden

#!/usr/bin/env python3
"""ATTENTION NLP"""
import tensorflow
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tensorflow.keras.layers.Layer):
    """ decode for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        """class constructor"""
        super(RNNDecoder, self).__init__()
        self.embedding = tensorflow.keras.layers.Embedding(vocab, embedding)
        self.gru = tensorflow.keras.layers.GRU(units=units, kernel_initializer='glorot_uni\
            form', return_sequences=True, return_state=True)
        self.F = tensorflow.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """public instance method"""
        context_vector, attention_weights = self.attention(
            s_prev, hidden_states)
        x = self.embedding(x)
        x = tensorflow.concat([tensorflow.expand_dims(
            context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tensorflow.reshape(output, (-1, output.shape[2]))
        x = self.F(output)
        return x, state

#!/usr/bin/env python3
"""ATTENTION NLP"""
import tensorflow


class SelfAttention(tensorflow.keras.layers.Layer):
    """calculate the attention for machine translation based
        on Dzmitry BahdanauJacobs, KyungHyun Cho and Yoshua Bengio
        conference paper  at ICLR 2015"""
    def __init__(self, units):
        """CLASS CONSTRUCTOR"""
        super(SelfAttention, self).__init__()
        self.W = tensorflow.keras.layers.Dense(units)
        self.U = tensorflow.keras.layers.Dense(units)
        self.V = tensorflow.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """calls function"""
        s_prev = tensorflow.expand_dims(s_prev, 1)
        e = self.V(tensorflow.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        a = tensorflow.nn.softmax(e, axis=1)
        c = a * hidden_states
        c = tensorflow.reduce_sum(c, axis=1)
        return c, a

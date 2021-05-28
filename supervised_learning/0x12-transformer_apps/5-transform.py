#!/usr/bin/env python3
"""TRANSFORMER APP"""
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer

def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """creates and trains a transformer model for machine translation of Pt/En"""

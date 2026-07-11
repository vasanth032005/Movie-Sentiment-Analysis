"""
====================================================
Transformer Encoder Demo
====================================================

Author : Vasanth

Simple Transformer Encoder Example

====================================================
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    MultiHeadAttention,
    LayerNormalization,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model

# ---------------------------------
# Parameters
# ---------------------------------

EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
SEQUENCE_LENGTH = 20

# ---------------------------------
# Input Layer
# ---------------------------------

inputs = Input(shape=(SEQUENCE_LENGTH, EMBED_DIM))

# ---------------------------------
# Multi Head Attention
# ---------------------------------

attention = MultiHeadAttention(
    num_heads=NUM_HEADS,
    key_dim=EMBED_DIM
)

attention_output = attention(inputs, inputs)

attention_output = Dropout(0.1)(attention_output)

x = LayerNormalization(epsilon=1e-6)(
    inputs + attention_output
)

# ---------------------------------
# Feed Forward Network
# ---------------------------------

ffn = Dense(FF_DIM, activation="relu")(x)

ffn = Dense(EMBED_DIM)(ffn)

outputs = LayerNormalization(epsilon=1e-6)(
    x + ffn
)

model = Model(inputs, outputs)

model.summary()
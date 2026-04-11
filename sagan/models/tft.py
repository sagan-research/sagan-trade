"""Temporal Fusion Transformer components (simplified)"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class VariableSelectionNetwork(layers.Layer):
    """Soft feature gating over the stock dimension."""

    def __init__(self, num_features: int, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.units = units
        self.gate = tf.keras.Sequential([
            layers.Dense(units, activation='tanh'),
            layers.Dense(num_features, activation='softmax'),
        ])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        weights = self.gate(tf.reduce_mean(inputs, axis=1))   # (B, n_stocks)
        weights = tf.expand_dims(weights, axis=1)              # (B, 1, n_stocks)
        return inputs * weights                                # (B, T, n_stocks)


class TemporalFusionBlock(layers.Layer):
    """Single multi-head self-attention + feed-forward block."""

    def __init__(
        self,
        head_dim: int = 32,
        num_heads: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_dim, dropout=dropout
        )
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(head_dim * num_heads),
        ])
        self.dropout = layers.Dropout(dropout)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        attn_out = self.attention(x, x)
        x = self.layernorm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_out))
        return x


def build_tft_action_model(
    window: int,
    n_stocks: int,
    head_dim: int = 32,
    num_heads: int = 4,
    ff_dim: int = 64,
    dropout: float = 0.1,
) -> Model:
    """Build the TFT-based single-output (logit) action model."""
    inp = layers.Input(shape=(window, n_stocks))
    selected = VariableSelectionNetwork(n_stocks, units=32)(inp)
    tft_out = TemporalFusionBlock(head_dim, num_heads, ff_dim, dropout)(selected)
    pooled = layers.GlobalAveragePooling1D()(tft_out)
    logit = layers.Dense(1)(pooled)
    return Model(inputs=inp, outputs=logit)

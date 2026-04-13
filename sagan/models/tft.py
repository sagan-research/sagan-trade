"""Temporal Fusion Transformer components (simplified)"""

import tensorflow as tf
from tensorflow.keras import layers, Model


@tf.keras.utils.register_keras_serializable(package="sagan")
class VariableSelectionNetwork(layers.Layer):
    """Soft feature gating over the stock dimension."""

    def __init__(self, num_features: int, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.units = units
        self.gate = tf.keras.Sequential([
            layers.Dense(units, activation='tanh', name='gate_dense_1'),
            layers.Dense(num_features, activation='softmax', name='gate_softmax'),
        ])

    def call(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # inputs shape: (B, T, n_stocks)
        weights = self.gate(tf.reduce_mean(inputs, axis=1))   # (B, n_stocks)
        weights_expanded = tf.expand_dims(weights, axis=1)    # (B, 1, n_stocks)
        gated = inputs * weights_expanded                      # (B, T, n_stocks)
        return gated, weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "units": self.units,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="sagan")
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
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


def build_tft_action_model(
    window: int,
    n_stocks: int,
    head_dim: int = 32,
    num_heads: int = 4,
    ff_dim: int = 64,
    dropout: float = 0.1,
) -> Model:
    """Build the TFT-based dual-output (logit, selection_weights) model."""
    inp = layers.Input(shape=(window, n_stocks), name="input_layer")
    selected, selection_weights = VariableSelectionNetwork(n_stocks, units=32, name="vsn")(inp)
    
    # Project features to match attention head dimensions for residual connection
    d_model = head_dim * num_heads
    projected = layers.Dense(d_model, name="feature_projection")(selected)
    
    tft_out = TemporalFusionBlock(head_dim, num_heads, ff_dim, dropout, name="tft_block")(projected)
    pooled = layers.GlobalAveragePooling1D(name="pooling")(tft_out)
    logit = layers.Dense(1, name="logit")(pooled)
    
    # Selection weights are for explainability/weighting, not supervised training
    selection_weights = layers.Activation('linear', name="selection_weights")(selection_weights)
    
    return Model(inputs=inp, outputs={"logit": logit, "selection_weights": selection_weights})

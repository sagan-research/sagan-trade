"""XAI-RL layer for regime detection and uncertainty-based override"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="sagan")
class XAIRLLayer(tf.keras.layers.Layer):
    """
    Computes regime uncertainty from prediction logits.

    Returns a dict with:
      - regime_uncertainty: 1 - max softmax probability
      - max_prob:           highest softmax probability
      - override:           bool mask where confidence < threshold
    """

    def __init__(self, threshold: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(
        self,
        attention_weights: tf.Tensor,
        prediction_logits: tf.Tensor,
    ) -> dict:
        probs = tf.nn.softmax(prediction_logits, axis=-1)
        max_prob = tf.reduce_max(probs, axis=-1)
        override = max_prob < self.threshold
        return {
            "regime_uncertainty": 1.0 - max_prob,
            "max_prob": max_prob,
            "override": override,
        }

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"threshold": self.threshold})
        return cfg

"""Physics-Informed Neural Network loss for mean reversion"""

import tensorflow as tf


def ou_process_residual(
    x: tf.Tensor,
    theta: float = 0.1,
    mu: float = 0.0,
    sigma: float = 0.02,
    dt: float = 1.0,
) -> tf.Tensor:
    """Penalty for probability deviating from 0.5 (mean‑reverting assumption)."""
    p = tf.nn.sigmoid(x)
    return tf.reduce_mean(tf.square(p - 0.5))


def pinn_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor, lambda_pinn: float = 0.01) -> tf.Tensor:
    """Binary cross-entropy + Ornstein–Uhlenbeck mean-reversion penalty."""
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
    bce = tf.reduce_mean(bce)
    penalty = ou_process_residual(y_pred_logits)
    return bce + lambda_pinn * penalty

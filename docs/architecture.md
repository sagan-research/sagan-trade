# Architecture

Sagan XAI combines three distinct technical innovations into a single
end-to-end training pipeline. This page gives a detailed mathematical and
architectural description of each component.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│   Raw OHLCV prices  →  pct_change  →  (N, window, n_stocks)    │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │   Variable Selection Network   │
                │   Softmax-gated attention over │
                │   the stock / feature axis     │
                └───────────────┬───────────────┘
                                │
                ┌───────────────▼───────────────┐
                │    Temporal Fusion Block       │
                │  MultiHeadAttention + FFN      │
                │  LayerNorm + Dropout           │
                └───────────────┬───────────────┘
                                │
                ┌───────────────▼───────────────┐
                │   GlobalAveragePooling1D       │
                └───┬───────────┬───────────┬───┘
                    │           │           │
              ┌─────▼──┐  ┌────▼───┐  ┌────▼───┐
              │ BUY    │  │ SELL   │  │ HOLD   │
              │ Dense  │  │ Dense  │  │ Dense  │
              │ (logit)│  │ (logit)│  │ (logit)│
              └─────┬──┘  └────┬───┘  └────┬───┘
                    └──────────┼───────────┘
                               │  softmax
                    ┌──────────▼───────────┐
                    │  LONG / SHORT /       │
                    │  NEUTRAL  +  probs   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   XAI-RL Override    │
                    │  max(p) < θ → flag   │
                    └──────────────────────┘
```

---

## 1. Variable Selection Network

The **Variable Selection Network (VSN)** learns to up-weight informative
stocks and down-weight noisy ones, reducing the curse of dimensionality in
high-dimensional baskets.

### Mechanism

Given an input tensor $\mathbf{X} \in \mathbb{R}^{B \times T \times N}$
(batch × time-steps × stocks):

1. Compute a context vector by averaging over the time dimension:
   $\mathbf{c} = \text{mean}_{t}(\mathbf{X}) \in \mathbb{R}^{B \times N}$

2. Pass through a two-layer gate:
   $\mathbf{w} = \text{softmax}(W_2 \tanh(W_1 \mathbf{c} + b_1) + b_2)$

3. Apply element-wise weight broadcast:
   $\tilde{\mathbf{X}} = \mathbf{X} \odot \mathbf{w}^{\top}$

The softmax constraint ensures weights sum to 1 and are interpretable as
*attention scores over the stock dimension*.

---

## 2. Temporal Fusion Block

The **Temporal Fusion Block** is a simplified variant of the
[Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) (Lim et al.,
2019), adapted for the univariate action-logit output used in Sagan.

### Components

| Layer | Purpose |
|---|---|
| `MultiHeadAttention` | Attend over historical return patterns |
| `LayerNormalization` (1) | Stabilise after attention residual |
| `Dense(ff_dim, relu) → Dense(head_dim × num_heads)` | Position-wise FFN |
| `LayerNormalization` (2) | Stabilise after FFN residual |
| `Dropout` | Regularisation before each residual add |

### Forward pass

$$
\begin{aligned}
\mathbf{a} &= \text{MultiHeadAttn}(\mathbf{X}, \mathbf{X}) \\
\mathbf{h}_1 &= \text{LayerNorm}(\mathbf{X} + \text{Dropout}(\mathbf{a})) \\
\mathbf{f} &= \text{FFN}(\mathbf{h}_1) \\
\mathbf{h}_2 &= \text{LayerNorm}(\mathbf{h}_1 + \text{Dropout}(\mathbf{f}))
\end{aligned}
$$

A `GlobalAveragePooling1D` layer then collapses the time dimension:
$\bar{\mathbf{h}} = \text{mean}_{t}(\mathbf{h}_2) \in \mathbb{R}^{B \times D}$

---

## 3. PINN Loss — Ornstein–Uhlenbeck Mean Reversion

The core insight of Sagan is encoding the **mean-reversion prior** directly
into the training objective via a Physics-Informed Neural Network (PINN) penalty.

### Ornstein–Uhlenbeck process

The OU process is the canonical continuous-time model for mean-reverting
dynamics:

$$
dS_t = \theta (\mu - S_t)\, dt + \sigma\, dW_t
$$

where $\theta > 0$ is the speed of reversion, $\mu$ is the long-run mean,
and $\sigma$ is volatility.

Under the OU prior, the *stationary distribution* of log-prices is
$\mathcal{N}(\mu, \sigma^2 / 2\theta)$. Translated to prediction space: the
probability of a *up* move should be close to 0.5 over long horizons.

### Sagan loss function

$$
\mathcal{L} = \underbrace{\frac{1}{N}\sum_i \text{BCE}(y_i, \hat{y}_i)}_{\text{supervised signal}}
+ \lambda \underbrace{\mathbb{E}\left[\left(\sigma(\hat{z}) - 0.5\right)^2\right]}_{\text{OU penalty}}
$$

where $\hat{z}$ is the raw logit and $\sigma(\cdot)$ is the sigmoid function.
The penalty is zero when the network is maximally uncertain (p = 0.5) and
grows as the model makes extreme predictions.

**Why this works:** The penalty acts as a Bayesian prior that prevents the
model from over-fitting to short-term noise, while the BCE term pulls it
toward genuine directional signals.

The parameter $\lambda$ (`pinn_lambda`) controls the trade-off. Higher
values impose stronger mean-reversion regularisation.

---

## 4. XAI-RL Override

The **XAI-RL override layer** is the explainability backbone of Sagan. It
monitors the softmax distribution of the ensemble output and raises an
`override` flag when the winning probability is below a threshold $\theta$.

### Regime uncertainty

$$
u = 1 - \max_k p_k
$$

where $p_k = \text{softmax}(\hat{z})_k$ is the probability of action $k$.
When $u$ is high, the ensemble is unsure about the current market regime.

### Override condition

$$
\text{override} = \mathbf{1}\left[\max_k p_k < \theta\right]
$$

The threshold $\theta$ is `xai_confidence_threshold` in
:class:`~sagan.config.SaganConfig` (default 0.6).

When `override = True`, downstream systems should treat the signal with extra
caution — for example, halving position size or requiring human confirmation.

---

## 5. Training & Label Generation

### Dataset construction

For a price matrix $P \in \mathbb{R}^{T \times N}$:

1. Compute daily returns: $r_t = \frac{P_t}{P_{t-1}} - 1$
2. Slide a window of size $w$ across the return series.
3. For each window $i$, compute the *forward return*:
   $\bar{r}_i = \frac{1}{h N} \sum_{j=1}^{h} \sum_{k=1}^{N} r_{i+w+j,k}$
4. Assign a one-hot label:
   - **BUY** if $\bar{r}_i > \delta$
   - **SELL** if $\bar{r}_i < -\delta$
   - **HOLD** otherwise

where $h$ = `horizon` and $\delta$ = `threshold`.

### Scaling

Inputs are standardised per-stock using a `StandardScaler` fit on the
training split only. The fitted scaler is saved alongside the model weights
and applied identically at inference time.

---

## 6. Validation Metric — Annualised Sharpe

After training, the ensemble is evaluated on the held-out 20 % validation
split using a *simulated strategy return*:

$$
r^{\text{strategy}}_t =
\begin{cases}
 r_t & \text{if action = BUY} \\
-r_t & \text{if action = SELL} \\
 0   & \text{if action = HOLD}
\end{cases}
$$

The annualised Sharpe ratio is then:

$$
\text{Sharpe} = \sqrt{252} \cdot \frac{\bar{r}^{\text{strategy}}}{\text{std}(r^{\text{strategy}})}
$$

This metric is stored in `metadata["val_sharpe"]` and can be inspected via
:func:`~sagan.registry.list_models`.

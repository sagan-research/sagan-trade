"""
Temporal Fusion Transformer (TFT) Implementation for Quantitative Finance.

Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
by Bryan Lim and Sercan Ö. Arık (Google Research, 2021).

This implementation is optimized for financial time series forecasting with:
- Multi-horizon prediction (intraday, daily, weekly)
- Interpretable attention mechanisms
- Static covariates (asset metadata)
- Time-varying known/unknown inputs
- Quantile outputs for uncertainty quantification
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""

    # Model dimensions
    hidden_size: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    num_encoder_steps: int = 168  # Lookback window (e.g., 1 week of hourly data)
    num_decoder_steps: int = 24  # Prediction horizon (e.g., 1 day ahead)

    # Feature dimensions
    num_static_vars: int = 10  # Static covariates (sector, market cap, etc.)
    num_time_varying_known: int = 5  # Known future inputs (calendar, holidays)
    num_time_varying_unknown: int = 20  # Unknown future inputs (lagged returns, vol)

    # Output
    output_size: int = 1  # Prediction dimension (e.g., return)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)  # Quantile outputs

    # Regularization
    use_cudnn_lstm: bool = True
    lstm_layers: int = 2

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert len(self.quantiles) > 0, "At least one quantile required"


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) - core building block of TFT."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
        context_size: int | None = None,
        return_gate: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.context_size = context_size
        self.return_gate = return_gate

        # Skip connection projection if dimensions differ
        self.skip_layer = (
            nn.Linear(input_size, self.output_size)
            if input_size != self.output_size
            else nn.Identity()
        )

        # Main network
        self.fc1 = nn.Linear(input_size + (context_size or 0), hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(self.output_size, self.output_size)
        self.layer_norm = nn.LayerNorm(self.output_size)
        self.elu = nn.ELU()

    def forward(self, x: Tensor, context: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x: Input tensor [..., input_size]
            context: Optional context tensor [..., context_size]
        Returns:
            output: [..., output_size]
            gate_weights: [..., output_size] if return_gate else None
        """
        # Add context if provided
        if context is not None:
            x = torch.cat([x, context], dim=-1)

        # Main path
        residual = self.skip_layer(x)
        hidden = self.elu(self.fc1(x))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)

        # Gating mechanism
        gate = torch.sigmoid(self.gate(output))
        gated_output = gate * output

        # Residual connection + LayerNorm
        final_output = self.layer_norm(gated_output + residual)

        if self.return_gate:
            return final_output, gate
        return final_output, None


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for automatic feature selection and weighting."""

    def __init__(
        self,
        num_inputs: int,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int | None = None,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Flatten all inputs for processing
        self.flatten = nn.Flatten(
            start_dim=-2
        )  # [..., num_inputs, input_size] -> [..., num_inputs * input_size]

        # GRN for variable selection weights
        self.grn = GatedResidualNetwork(
            input_size=num_inputs * input_size,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout=dropout,
            context_size=context_size,
        )

        # Individual GRNs for each variable
        self.variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                    context_size=context_size,
                )
                for _ in range(num_inputs)
            ]
        )

    def forward(self, x: Tensor, context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor [..., num_inputs, input_size]
            context: Optional context tensor [..., context_size]
        Returns:
            combined: [..., hidden_size] - weighted combination
            weights: [..., num_inputs] - selection weights (softmax)
        """
        _batch_shape = x.shape[:-2]
        num_inputs = x.shape[-2]

        # Flatten for weight computation
        flat_x = x.flatten(start_dim=-2)  # [..., num_inputs * input_size]
        weights, _ = self.grn(flat_x, context)  # [..., num_inputs]
        weights = F.softmax(weights, dim=-1)  # [..., num_inputs]

        # Process each variable through its GRN
        var_outputs = []
        for i in range(num_inputs):
            var_x = x[..., i, :]  # [..., input_size]
            var_out, _ = self.variable_grns[i](var_x, context)
            var_outputs.append(var_out)  # [..., hidden_size]

        # Stack: [..., num_inputs, hidden_size]
        var_outputs = torch.stack(var_outputs, dim=-2)

        # Weighted combination
        weights_expanded = weights.unsqueeze(-1)  # [..., num_inputs, 1]
        combined = (weights_expanded * var_outputs).sum(dim=-2)  # [..., hidden_size]

        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretable attention weights."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            query: [..., seq_len_q, hidden_size]
            key: [..., seq_len_k, hidden_size]
            value: [..., seq_len_v, hidden_size]
            mask: Optional mask [..., seq_len_q, seq_len_k]
        Returns:
            output: [..., seq_len_q, hidden_size]
            attention_weights: [..., num_heads, seq_len_q, seq_len_k]
        """
        batch_shape = query.shape[:-2]
        seq_len_q = query.shape[-2]
        seq_len_k = key.shape[-2]

        # Project and reshape for multi-head
        q = (
            self.q_proj(query)
            .view(*batch_shape, seq_len_q, self.num_heads, self.head_dim)
            .transpose(-3, -2)
        )
        k = (
            self.k_proj(key)
            .view(*batch_shape, seq_len_k, self.num_heads, self.head_dim)
            .transpose(-3, -2)
        )
        v = (
            self.v_proj(value)
            .view(*batch_shape, seq_len_k, self.num_heads, self.head_dim)
            .transpose(-3, -2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [..., num_heads, seq_len_q, head_dim]

        # Reshape and project
        attn_output = (
            attn_output.transpose(-3, -2)
            .contiguous()
            .view(*batch_shape, seq_len_q, self.hidden_size)
        )
        output = self.out_proj(attn_output)

        return output, attn_weights


class StaticCovariateEncoders(nn.Module):
    """Encodes static covariates into context vectors."""

    def __init__(
        self,
        num_static_vars: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_static_vars = num_static_vars
        self.hidden_size = hidden_size

        # Variable selection for static covariates
        self.variable_selection = VariableSelectionNetwork(
            num_inputs=num_static_vars,
            input_size=1,  # Each static var is scalar
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=None,
        )

        # Context GRNs
        self.context_variable_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )
        self.context_enrichment_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )
        self.context_state_h_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )
        self.context_state_c_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

    def forward(self, static_vars: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            static_vars: [batch_size, num_static_vars] or [batch_size, num_static_vars, 1]
        Returns:
            static_context_variable: [batch_size, num_static_vars] - selection weights
            static_context_enrichment: [batch_size, hidden_size]
            static_context_state_h: [batch_size, hidden_size]
            static_context_state_c: [batch_size, hidden_size]
        """
        # Ensure 3D: [batch, num_vars, 1]
        if static_vars.dim() == 2:
            static_vars = static_vars.unsqueeze(-1)

        # Variable selection
        static_context, static_weights = self.variable_selection(static_vars)

        # Context vectors for different purposes
        context_enrichment = self.context_enrichment_grn(static_context)[0]
        context_state_h = self.context_state_h_grn(static_context)[0]
        context_state_c = self.context_state_c_grn(static_context)[0]

        return static_weights, context_enrichment, context_state_h, context_state_c


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for Multi-Horizon Time Series Forecasting.

    Architecture components:
    1. Static Covariate Encoders
    2. Variable Selection Networks (encoder & decoder)
    3. LSTM Encoder/Decoder for local processing
    4. Interpretable Multi-Head Attention (static & temporal)
    5. Position-wise Feed-forward (GRN)
    6. Output Quantile Projection
    """

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_encoder_steps = config.num_encoder_steps
        self.num_decoder_steps = config.num_decoder_steps
        self.output_size = config.output_size
        self.quantiles = config.quantiles

        # Static covariate encoders
        self.static_encoders = StaticCovariateEncoders(
            num_static_vars=config.num_static_vars,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        )

        # Variable selection networks
        self.encoder_variable_selection = VariableSelectionNetwork(
            num_inputs=config.num_time_varying_unknown,
            input_size=1,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size,
        )
        self.decoder_variable_selection = VariableSelectionNetwork(
            num_inputs=config.num_time_varying_known + config.num_time_varying_unknown,
            input_size=1,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size,
        )

        # LSTM Encoder/Decoder for local processing
        lstm_class = nn.LSTM if config.use_cudnn_lstm else nn.LSTM
        self.encoder_lstm = lstm_class(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            bidirectional=False,
        )
        self.decoder_lstm = lstm_class(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # Post-LSTM GRNs
        self.encoder_post_lstm_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size,
        )
        self.decoder_post_lstm_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size,
        )

        # Static enrichment for temporal features
        self.static_enrichment_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size,
        )

        # Temporal Self-Attention (Interpretable Multi-Head Attention)
        self.temporal_attention = InterpretableMultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size,
        )

        # Position-wise Feed-forward (final GRN before output)
        self.position_wise_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size,
        )

        # Output projection for quantiles
        num_quantiles = len(config.quantiles)
        self.output_projection = nn.Linear(config.hidden_size, config.output_size * num_quantiles)

    def forward(
        self,
        static_inputs: Tensor,
        encoder_inputs: Tensor,  # [batch, enc_len, num_unknown]
        decoder_inputs: Tensor,  # [batch, dec_len, num_known + num_unknown]
    ) -> dict[str, Tensor]:
        """
        Forward pass of TFT.

        Args:
            static_inputs: [batch_size, num_static_vars]
            encoder_inputs: [batch_size, num_encoder_steps, num_time_varying_unknown]
            decoder_inputs: [batch_size, num_decoder_steps, num_time_varying_known + num_time_varying_unknown]

        Returns:
            Dictionary containing:
                - predictions: [batch_size, num_decoder_steps, num_quantiles]
                - attention_weights: Temporal attention weights for interpretability
                - variable_selection_weights: Encoder/Decoder variable importance
                - static_weights: Static variable importance
        """
        batch_size = static_inputs.shape[0]
        device = static_inputs.device

        # 1. Static Covariate Encoding
        (
            static_weights,
            static_context_enrichment,
            static_context_state_h,
            static_context_state_c,
        ) = self.static_encoders(static_inputs)

        # Initialize LSTM states from static context
        encoder_h0 = static_context_state_h.unsqueeze(0).repeat(self.config.lstm_layers, 1, 1)
        encoder_c0 = static_context_state_c.unsqueeze(0).repeat(self.config.lstm_layers, 1, 1)
        decoder_h0 = static_context_state_h.unsqueeze(0).repeat(self.config.lstm_layers, 1, 1)
        decoder_c0 = static_context_state_c.unsqueeze(0).repeat(self.config.lstm_layers, 1, 1)

        # 2. Encoder Variable Selection
        # encoder_inputs: [batch, enc_len, num_unknown] -> [batch, enc_len, num_unknown, 1]
        enc_vars = encoder_inputs.unsqueeze(-1)
        enc_selected, enc_var_weights = self.encoder_variable_selection(
            enc_vars, context=static_context_enrichment
        )  # [batch, enc_len, hidden_size]

        # 3. Encoder LSTM
        enc_lstm_out, _ = self.encoder_lstm(enc_selected, (encoder_h0, encoder_c0))

        # 4. Post-LSTM GRN with static context
        enc_processed, _ = self.encoder_post_lstm_grn(
            enc_lstm_out, context=static_context_enrichment
        )

        # 5. Decoder Variable Selection
        # decoder_inputs: [batch, dec_len, num_known + num_unknown] -> [batch, dec_len, num_vars, 1]
        dec_vars = decoder_inputs.unsqueeze(-1)
        dec_selected, dec_var_weights = self.decoder_variable_selection(
            dec_vars, context=static_context_enrichment
        )  # [batch, dec_len, hidden_size]

        # 6. Decoder LSTM
        dec_lstm_out, _ = self.decoder_lstm(dec_selected, (decoder_h0, decoder_c0))

        # 7. Post-LSTM GRN
        dec_processed, _ = self.decoder_post_lstm_grn(
            dec_lstm_out, context=static_context_enrichment
        )

        # 8. Static Enrichment for Temporal Features
        # Combine encoder and decoder for attention
        temporal_features = torch.cat([enc_processed, dec_processed], dim=1)
        enriched, _ = self.static_enrichment_grn(
            temporal_features, context=static_context_enrichment
        )

        # Split back
        enc_len = self.num_encoder_steps
        _enc_enriched = enriched[:, :enc_len]
        dec_enriched = enriched[:, enc_len:]

        # 9. Temporal Self-Attention (Decoder attends to Encoder + Decoder)
        # Use causal mask for decoder self-attention
        dec_seq_len = dec_enriched.shape[1]
        causal_mask = torch.tril(
            torch.ones(dec_seq_len, enc_len + dec_seq_len, device=device)
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, dec_len, enc_len+dec_len]

        attn_out, attn_weights = self.temporal_attention(
            query=dec_enriched,
            key=enriched,
            value=enriched,
            mask=causal_mask,
        )

        # 10. Post-Attention GRN
        post_attn, _ = self.post_attention_grn(attn_out, context=static_context_enrichment)

        # 11. Position-wise Feed-forward
        final_features, _ = self.position_wise_grn(post_attn, context=static_context_enrichment)

        # 12. Output Projection (Quantiles)
        # final_features: [batch, dec_len, hidden_size]
        output = self.output_projection(
            final_features
        )  # [batch, dec_len, output_size * num_quantiles]
        num_quantiles = len(self.quantiles)
        output = output.view(batch_size, self.num_decoder_steps, self.output_size, num_quantiles)
        predictions = output.permute(0, 1, 3, 2)  # [batch, dec_len, num_quantiles, output_size]

        return {
            "predictions": predictions,
            "attention_weights": attn_weights,
            "encoder_variable_weights": enc_var_weights,
            "decoder_variable_weights": dec_var_weights,
            "static_weights": static_weights,
        }

    def predict_quantiles(
        self,
        static_inputs: Tensor,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
    ) -> Tensor:
        """Get quantile predictions only."""
        outputs = self.forward(static_inputs, encoder_inputs, decoder_inputs)
        return outputs["predictions"]

    def predict_median(
        self,
        static_inputs: Tensor,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
    ) -> Tensor:
        """Get median (0.5 quantile) predictions."""
        outputs = self.forward(static_inputs, encoder_inputs, decoder_inputs)
        predictions = outputs["predictions"]  # [batch, dec_len, num_quantiles, output_size]
        # Find index of median quantile (0.5)
        median_idx = (
            self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        )
        return predictions[:, :, median_idx, :]


def create_tft_model(
    num_static_vars: int = 10,
    num_time_varying_known: int = 5,
    num_time_varying_unknown: int = 20,
    hidden_size: int = 256,
    num_heads: int = 4,
    num_encoder_steps: int = 168,
    num_decoder_steps: int = 24,
    output_size: int = 1,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    dropout: float = 0.1,
) -> TemporalFusionTransformer:
    """Factory function to create TFT model with default configuration."""
    config = TFTConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=dropout,
        num_encoder_steps=num_encoder_steps,
        num_decoder_steps=num_decoder_steps,
        num_static_vars=num_static_vars,
        num_time_varying_known=num_time_varying_known,
        num_time_varying_unknown=num_time_varying_unknown,
        output_size=output_size,
        quantiles=quantiles,
    )
    return TemporalFusionTransformer(config)


if __name__ == "__main__":
    # Test the model
    config = TFTConfig(
        hidden_size=128,
        num_heads=4,
        num_encoder_steps=48,
        num_decoder_steps=12,
        num_static_vars=5,
        num_time_varying_known=3,
        num_time_varying_unknown=10,
        output_size=1,
        quantiles=(0.1, 0.5, 0.9),
        dropout=0.1,
    )

    model = TemporalFusionTransformer(config)

    batch_size = 32
    static_inputs = torch.randn(batch_size, 5)
    encoder_inputs = torch.randn(batch_size, 48, 10)
    decoder_inputs = torch.randn(batch_size, 12, 13)  # 3 known + 10 unknown

    outputs = model(static_inputs, encoder_inputs, decoder_inputs)

    print("Model created successfully!")
    print(f"Predictions shape: {outputs['predictions'].shape}")  # [32, 12, 3, 1]
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    print(f"Encoder var weights shape: {outputs['encoder_variable_weights'].shape}")
    print(f"Decoder var weights shape: {outputs['decoder_variable_weights'].shape}")
    print(f"Static weights shape: {outputs['static_weights'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

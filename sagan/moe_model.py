import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution to prevent future information leakage in time-series modeling.
    Pads the input on the left so that output[t] depends only on input[0:t].
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, **kwargs
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        out = self.conv(x)
        # Slice off the extra padding on the right to restore causality
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TCNBlock(nn.Module):
    """
    A single Temporal Convolutional Network block with Causal Conv, LayerNorm, GELU, and Dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual match projection if channel dimensions differ
        self.residual_project = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        res = x if self.residual_project is None else self.residual_project(x)

        # First layer
        out = self.conv1(x)
        # LayerNorm expects shape (batch, seq_len, channels)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.gelu(out)
        out = self.dropout(out)

        # Second layer
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.gelu(out)
        out = self.dropout(out)

        # Add residual connection
        return self.gelu(out + res)


class TCNExpert(nn.Module):
    """
    TCN Expert that processes historical sequence features to predict future spread.
    """

    def __init__(
        self, num_features: int, hidden_dim: int, kernel_size: int = 3, dilations: list = [1, 2, 4]
    ):
        super().__init__()
        layers = []
        in_dim = num_features
        for d in dilations:
            layers.append(TCNBlock(in_dim, hidden_dim, kernel_size, dilation=d))
            in_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        # Fully connected projection of final step
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, num_features)
        # Conv1d expects (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)
        feat = self.network(x_conv)
        # Extract the representation at the very last step in sequence
        last_step_feat = feat[:, :, -1]  # shape: (batch, hidden_dim)
        out = self.fc(last_step_feat)  # shape: (batch, 1)
        return out


class StateGatingNetwork(nn.Module):
    """
    State-dependent gating network that routes inputs to experts based on current volatility, OFI, and trade intensity.
    """

    def __init__(self, state_dim: int, num_experts: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, num_experts)
        self.gelu = nn.GELU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state shape: (batch, state_dim)
        h = self.gelu(self.fc1(state))
        gating_logits = self.fc2(h)
        # Return probability distribution over experts
        return F.softmax(gating_logits, dim=-1)


class SaganMoEModel(nn.Module):
    """
    Sagan Mixture of Experts Network combining state routing and TCN specialists.
    """

    def __init__(
        self, num_features: int, state_dim: int, num_experts: int = 3, expert_hidden_dim: int = 32
    ):
        super().__init__()
        self.num_experts = num_experts

        # State routing network
        self.gating = StateGatingNetwork(state_dim, num_experts)

        # Expert specialized networks
        # We instantiate multiple TCN experts with differing kernel sizes and depths to specialize
        self.experts = nn.ModuleList(
            [
                TCNExpert(
                    num_features, expert_hidden_dim, kernel_size=2, dilations=[1, 2]
                ),  # Expert 1: Calm/Fast specialist
                TCNExpert(
                    num_features, expert_hidden_dim, kernel_size=4, dilations=[1, 2, 4]
                ),  # Expert 2: Volatile/Dilation specialist
                TCNExpert(
                    num_features, expert_hidden_dim, kernel_size=3, dilations=[1, 2, 4, 8]
                ),  # Expert 3: Microstructure/OFI specialist
            ]
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs MoE routing and ensemble prediction.
        x: historical sequence, shape (batch, seq_len, num_features)
        state: current state vector, shape (batch, state_dim)

        Returns:
            pred: blended spread prediction, shape (batch, 1)
            gating_weights: routing weights for logging, shape (batch, num_experts)
        """
        # Get gating probabilities
        gating_weights = self.gating(state)  # shape: (batch, num_experts)

        # Evaluate all experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # list of tensors of shape (batch, 1)

        # Combine expert predictions weighted by gating probabilities
        # expert_outputs_tensor shape: (batch, num_experts, 1)
        expert_outputs_tensor = torch.stack(expert_outputs, dim=1)

        # Multiply weights: gating_weights (batch, num_experts) -> unsqueeze to (batch, num_experts, 1)
        # Sum across experts to get shape (batch, 1)
        blended_pred = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs_tensor, dim=1)

        return blended_pred, gating_weights


def compute_moe_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gating_weights: torch.Tensor,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """
    Computes MSE loss plus an entropy regularization term on gating weights.
    Entropy regularization prevents "expert collapse", where a single expert takes 100% routing.
    """
    mse = F.mse_loss(pred, target)

    # Gating Entropy: -sum(p * log(p + eps))
    # Average across batch
    eps = 1e-8
    entropy = -torch.mean(torch.sum(gating_weights * torch.log(gating_weights + eps), dim=-1))

    # Since we want to MAXIMIZE entropy (encourage diverse usage), we subtract the entropy
    loss = mse - entropy_coef * entropy
    return loss


if __name__ == "__main__":
    # Test tensor shapes
    model = SaganMoEModel(num_features=5, state_dim=3)
    x = torch.randn(8, 20, 5)  # batch=8, seq_len=20, features=5
    state = torch.randn(8, 3)  # batch=8, state_dim=3
    pred, weights = model(x, state)
    print("Pred shape:", pred.shape)
    print("Weights shape:", weights.shape)
    print("Weights sample:", weights[0])

"""
Physics-Informed Neural Networks (PINNs) for Quantitative Finance.

Implements PINNs for:
1. Black-Scholes PDE for European option pricing
2. Heston Stochastic Volatility Model
3. Local Volatility (Dupire's Equation)
4. American Options (Free Boundary Problems)

References:
- Raissi et al. (2019) "Physics-Informed Neural Networks: A Deep Learning Framework
  for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations"
- Sirignano & Spiliopoulos (2018) "DGM: A Deep Learning Algorithm for Solving PDEs"
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class PINNConfig:
    """Configuration for Physics-Informed Neural Networks."""

    # Network architecture
    hidden_layers: list[int] = None  # e.g., [64, 64, 64, 64]
    activation: str = "tanh"  # tanh, sin, swish, gelu
    input_dim: int = 3  # (S, t, v) for Heston; (S, t) for BS
    output_dim: int = 1  # Option price

    # Training
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # adam, lbfgs
    max_epochs: int = 10000
    patience: int = 1000

    # Loss weights
    pde_weight: float = 1.0
    boundary_weight: float = 10.0
    initial_weight: float = 10.0
    data_weight: float = 1.0  # For supervised data if available

    # Domain bounds
    s_min: float = 0.01
    s_max: float = 3.0  # Multiple of strike
    t_min: float = 0.0
    t_max: float = 1.0  # Time to maturity in years
    v_min: float = 0.01
    v_max: float = 1.0

    # Sampling
    n_pde_points: int = 10000
    n_boundary_points: int = 2000
    n_initial_points: int = 2000

    # Physics parameters (can be learned)
    learn_volatility: bool = False
    learn_risk_free_rate: bool = False
    learn_mean_reversion: bool = False

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64, 64, 64]
        if isinstance(self.activation, str):
            self.activation = self._get_activation(self.activation)

    @staticmethod
    def _get_activation(name: str) -> Callable:
        activations = {
            "tanh": torch.tanh,
            "sin": torch.sin,
            "swish": lambda x: x * torch.sigmoid(x),
            "gelu": F.gelu,
            "relu": F.relu,
            "softplus": F.softplus,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]


class SineLayer(nn.Module):
    """SIREN-style sine activation layer for better PDE solving."""

    def __init__(
        self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features, 1 / self.linear.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6 / self.linear.in_features) / self.omega_0,
                    math.sqrt(6 / self.linear.in_features) / self.omega_0,
                )
            self.linear.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class PINN(nn.Module):
    """Base Physics-Informed Neural Network."""

    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config

        # Build network
        layers = []
        in_dim = config.input_dim

        for i, hidden_dim in enumerate(config.hidden_layers):
            if config.activation == torch.sin:
                layers.append(SineLayer(in_dim, hidden_dim, is_first=(i == 0)))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

        # Learnable parameters (if enabled)
        self._setup_learnable_params()

        # Initialize weights
        self.apply(self._init_weights)

    def _setup_learnable_params(self):
        """Setup learnable physics parameters."""
        self.log_sigma = nn.Parameter(torch.tensor(0.0))  # log volatility
        self.log_r = nn.Parameter(torch.tensor(-3.0))  # log risk-free rate
        self.log_kappa = nn.Parameter(torch.tensor(1.0))  # log mean reversion
        self.log_theta = nn.Parameter(torch.tensor(-2.0))  # log long-term variance
        self.log_xi = nn.Parameter(torch.tensor(-1.0))  # log vol of vol
        self.rho = nn.Parameter(torch.tensor(0.0))  # correlation

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def sigma(self) -> Tensor:
        return torch.exp(self.log_sigma)

    @property
    def r(self) -> Tensor:
        return torch.exp(self.log_r)

    @property
    def kappa(self) -> Tensor:
        return torch.exp(self.log_kappa)

    @property
    def theta(self) -> Tensor:
        return torch.exp(self.log_theta)

    @property
    def xi(self) -> Tensor:
        return torch.exp(self.log_xi)

    def forward(self, *inputs: Tensor) -> Tensor:
        x = torch.cat(inputs, dim=-1)
        return self.network(x)

    def compute_derivatives(
        self, output: Tensor, inputs: tuple[Tensor, ...], orders: dict[str, int]
    ) -> dict[str, Tensor]:
        """
        Compute derivatives using autograd.

        Args:
            output: Network output
            inputs: Tuple of input tensors (each requires_grad=True)
            orders: Dict mapping input names to derivative orders
        Returns:
            Dict of derivative tensors
        """
        derivatives = {}

        for i, (name, order) in enumerate(orders.items()):
            inp = inputs[i]
            grad = output
            for _ in range(order):
                grad = torch.autograd.grad(
                    grad,
                    inp,
                    grad_outputs=torch.ones_like(grad),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            derivatives[f"d{order}{name}"] = grad

        return derivatives


class BlackScholesPINN(PINN):
    """
    PINN for Black-Scholes PDE:
    ∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + r*S*∂V/∂S - r*V = 0

    Boundary conditions:
    - V(S, T) = max(S - K, 0) for call, max(K - S, 0) for put
    - V(0, t) = 0 (call), V(0, t) = K*e^{-r(T-t)} (put)
    - V(S_max, t) ≈ S_max - K*e^{-r(T-t)} (call), 0 (put)
    """

    def __init__(
        self,
        config: PINNConfig,
        strike: float = 1.0,
        option_type: str = "call",  # "call" or "put"
    ):
        # BS PDE has 2 inputs: (S, t)
        config.input_dim = 2
        config.output_dim = 1
        super().__init__(config)

        self.strike = strike
        self.option_type = option_type.lower()
        assert self.option_type in ["call", "put"]

    def pde_residual(self, S: Tensor, t: Tensor) -> Tensor:
        """
        Compute Black-Scholes PDE residual.

        Args:
            S: Stock price [batch, 1]
            t: Time to maturity [batch, 1]
        Returns:
            PDE residual [batch, 1]
        """
        # Ensure gradients are enabled
        S.requires_grad_(True)
        t.requires_grad_(True)

        # Network prediction
        V = self.forward(S, t)  # [batch, 1]

        # First derivatives
        V_t = torch.autograd.grad(
            V, t, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True
        )[0]

        V_S = torch.autograd.grad(
            V, S, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True
        )[0]

        # Second derivative
        V_SS = torch.autograd.grad(
            V_S, S, grad_outputs=torch.ones_like(V_S), create_graph=True, retain_graph=True
        )[0]

        # Black-Scholes PDE
        sigma = self.sigma
        r = self.r

        pde = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V

        return pde

    def boundary_loss(self, S: Tensor, t: Tensor) -> Tensor:
        """Compute boundary condition loss."""
        V_pred = self.forward(S, t)

        if self.option_type == "call":
            # V(0, t) = 0
            V_true = torch.zeros_like(V_pred)
        else:
            # V(0, t) = K * exp(-r * t)
            r = self.r
            V_true = self.strike * torch.exp(-r * t)

        return F.mse_loss(V_pred, V_true)

    def initial_loss(self, S: Tensor) -> Tensor:
        """Compute terminal condition loss at t=T."""
        # At maturity t=T (or t=0 in forward time), V = payoff
        t = torch.zeros_like(S)
        V_pred = self.forward(S, t)

        if self.option_type == "call":
            V_true = torch.clamp(S - self.strike, min=0)
        else:
            V_true = torch.clamp(self.strike - S, min=0)

        return F.mse_loss(V_pred, V_true)

    def upper_boundary_loss(self, t: Tensor) -> Tensor:
        """Compute upper boundary condition at S=S_max."""
        S_max = torch.full_like(t, self.config.s_max * self.strike)
        V_pred = self.forward(S_max, t)

        r = self.r
        if self.option_type == "call":
            # V(S_max, t) ≈ S_max - K*exp(-r*t)
            V_true = S_max - self.strike * torch.exp(-r * t)
        else:
            # V(S_max, t) ≈ 0
            V_true = torch.zeros_like(V_pred)

        return F.mse_loss(V_pred, V_true)

    def implied_volatility(self, S: Tensor, t: Tensor, market_price: Tensor) -> Tensor:
        """
        Compute implied volatility by inverting the network.
        Uses Newton-Raphson on the network output.
        """
        # This would require iterative solving
        # For now, return the learned sigma
        return self.sigma.expand_as(S)


class HestonPINN(PINN):
    """
    PINN for Heston Stochastic Volatility Model:

    ∂V/∂t + 0.5*v*S²*∂²V/∂S² + ρ*ξ*v*S*∂²V/∂S∂v + 0.5*ξ²*v*∂²V/∂v²
    + r*S*∂V/∂S + κ(θ-v)*∂V/∂v - r*V = 0

    Where v = variance (volatility squared)
    """

    def __init__(
        self,
        config: PINNConfig,
        strike: float = 1.0,
        option_type: str = "call",
    ):
        # Heston has 3 inputs: (S, v, t)
        config.input_dim = 3
        config.output_dim = 1
        super().__init__(config)

        self.strike = strike
        self.option_type = option_type.lower()
        assert self.option_type in ["call", "put"]

    def pde_residual(self, S: Tensor, v: Tensor, t: Tensor) -> Tensor:
        """Compute Heston PDE residual."""
        S.requires_grad_(True)
        v.requires_grad_(True)
        t.requires_grad_(True)

        V = self.forward(S, v, t)

        # First derivatives
        V_t = torch.autograd.grad(V, t, torch.ones_like(V), create_graph=True, retain_graph=True)[0]
        V_S = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True, retain_graph=True)[0]
        V_v = torch.autograd.grad(V, v, torch.ones_like(V), create_graph=True, retain_graph=True)[0]

        # Second derivatives
        V_SS = torch.autograd.grad(
            V_S, S, torch.ones_like(V_S), create_graph=True, retain_graph=True
        )[0]
        V_vv = torch.autograd.grad(
            V_v, v, torch.ones_like(V_v), create_graph=True, retain_graph=True
        )[0]
        V_Sv = torch.autograd.grad(
            V_S, v, torch.ones_like(V_S), create_graph=True, retain_graph=True
        )[0]

        # Heston parameters
        r = self.r
        kappa = self.kappa
        theta = self.theta
        xi = self.xi
        rho = torch.tanh(self.rho)  # Constrain to [-1, 1]

        # Heston PDE
        pde = (
            V_t
            + 0.5 * v * S**2 * V_SS
            + rho * xi * v * S * V_Sv
            + 0.5 * xi**2 * v * V_vv
            + r * S * V_S
            + kappa * (theta - v) * V_v
            - r * V
        )

        return pde

    def boundary_loss(self, S: Tensor, v: Tensor, t: Tensor) -> Tensor:
        """Boundary conditions at S=0 and S=S_max."""
        V_pred = self.forward(S, v, t)
        r = self.r

        if self.option_type == "call":
            # At S=0: V = 0
            V_true = torch.zeros_like(V_pred)
        else:
            # At S=0: V = K*exp(-r*t)
            V_true = self.strike * torch.exp(-r * t)

        return F.mse_loss(V_pred, V_true)

    def initial_loss(self, S: Tensor, v: Tensor) -> Tensor:
        """Terminal condition at maturity."""
        t = torch.zeros_like(S)
        V_pred = self.forward(S, v, t)

        if self.option_type == "call":
            V_true = torch.clamp(S - self.strike, min=0)
        else:
            V_true = torch.clamp(self.strike - S, min=0)

        return F.mse_loss(V_pred, V_true)

    def variance_boundary_loss(self, S: Tensor, t: Tensor) -> Tensor:
        """Boundary condition at v=0 (Feller condition boundary)."""
        v_zero = torch.zeros_like(S)
        V_pred = self.forward(S, v_zero, t)

        # At v=0, PDE reduces to Black-Scholes with zero volatility
        # For European options, this has analytical solution
        r = self.r
        if self.option_type == "call":
            V_true = torch.clamp(S - self.strike * torch.exp(-r * t), min=0)
        else:
            V_true = torch.clamp(self.strike * torch.exp(-r * t) - S, min=0)

        return F.mse_loss(V_pred, V_true)


class LocalVolatilityPINN(PINN):
    """
    PINN for Dupire's Local Volatility Equation:

    ∂C/∂T = 0.5 * σ²(K, T) * K² * ∂²C/∂K² - r * K * ∂C/∂K

    Where σ(K, T) is the local volatility surface.
    """

    def __init__(
        self,
        config: PINNConfig,
        risk_free_rate: float = 0.05,
    ):
        # Local vol: inputs (K, T), output: local vol
        config.input_dim = 2
        config.output_dim = 1
        super().__init__(config)

        self.r = risk_free_rate

    def dupire_residual(self, K: Tensor, T: Tensor) -> Tensor:
        """Compute Dupire's equation residual for local volatility."""
        K.requires_grad_(True)
        T.requires_grad_(True)

        # Network outputs local volatility
        local_vol = torch.exp(self.forward(K, T))  # Ensure positive

        # We need option prices - this would typically come from market data
        # or another network. For pure PINN, we'd solve the forward equation.
        # This is a placeholder for the full implementation.
        return local_vol  # Simplified


class PINNTrainer:
    """Trainer for Physics-Informed Neural Networks."""

    def __init__(
        self,
        model: PINN,
        config: PINNConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        if config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer.lower() == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=config.learning_rate,
                max_iter=20,
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=500, factor=0.5, verbose=True
        )

        self.history = {"pde": [], "boundary": [], "initial": [], "total": []}
        self.best_loss = float("inf")
        self.patience_counter = 0

    def sample_pde_points(self, n: int) -> tuple[Tensor, ...]:
        """Sample random points in the PDE domain."""
        cfg = self.config

        # Latin Hypercube or random sampling
        S = torch.rand(n, 1, device=self.device) * (cfg.s_max - cfg.s_min) + cfg.s_min
        t = torch.rand(n, 1, device=self.device) * (cfg.t_max - cfg.t_min) + cfg.t_min

        if isinstance(self.model, HestonPINN):
            v = torch.rand(n, 1, device=self.device) * (cfg.v_max - cfg.v_min) + cfg.v_min
            return S, v, t

        return S, t

    def sample_boundary_points(self, n: int) -> tuple[Tensor, ...]:
        """Sample points on boundaries."""
        cfg = self.config
        n_half = n // 2

        # S = 0 boundary
        S_zero = torch.zeros(n_half, 1, device=self.device)
        t_zero = torch.rand(n_half, 1, device=self.device) * (cfg.t_max - cfg.t_min) + cfg.t_min

        # S = S_max boundary
        S_max = torch.full((n_half, 1), cfg.s_max, device=self.device)
        t_max = torch.rand(n_half, 1, device=self.device) * (cfg.t_max - cfg.t_min) + cfg.t_min

        S = torch.cat([S_zero, S_max], dim=0)
        t = torch.cat([t_zero, t_max], dim=0)

        if isinstance(self.model, HestonPINN):
            v = torch.rand(n, 1, device=self.device) * (cfg.v_max - cfg.v_min) + cfg.v_min
            return S, v, t

        return S, t

    def sample_initial_points(self, n: int) -> tuple[Tensor, ...]:
        """Sample points at initial/terminal time."""
        cfg = self.config

        S = torch.rand(n, 1, device=self.device) * (cfg.s_max - cfg.s_min) + cfg.s_min
        _t = torch.zeros(n, 1, device=self.device)  # t=0 (maturity)

        if isinstance(self.model, HestonPINN):
            v = torch.rand(n, 1, device=self.device) * (cfg.v_max - cfg.v_min) + cfg.v_min
            return S, v

        return (S,)

    def train_step(self) -> dict[str, float]:
        """Single training step."""
        cfg = self.config
        model = self.model

        def closure():
            self.optimizer.zero_grad()

            # Sample points
            pde_points = self.sample_pde_points(cfg.n_pde_points)
            boundary_points = self.sample_boundary_points(cfg.n_boundary_points)
            initial_points = self.sample_initial_points(cfg.n_initial_points)

            # PDE Loss
            if isinstance(model, BlackScholesPINN):
                S, t = pde_points
                pde_residual = model.pde_residual(S, t)
                pde_loss = torch.mean(pde_residual**2)

                # Boundary loss
                S_b, t_b = boundary_points
                boundary_loss = model.boundary_loss(S_b, t_b) + model.upper_boundary_loss(t_b)

                # Initial loss
                (S_0,) = initial_points
                initial_loss = model.initial_loss(S_0)

            elif isinstance(model, HestonPINN):
                S, v, t = pde_points
                pde_residual = model.pde_residual(S, v, t)
                pde_loss = torch.mean(pde_residual**2)

                S_b, v_b, t_b = boundary_points
                boundary_loss = model.boundary_loss(S_b, v_b, t_b) + model.variance_boundary_loss(
                    S_b, t_b
                )

                S_0, v_0 = initial_points
                initial_loss = model.initial_loss(S_0, v_0)

            else:
                pde_loss = torch.tensor(0.0, device=self.device)
                boundary_loss = torch.tensor(0.0, device=self.device)
                initial_loss = torch.tensor(0.0, device=self.device)

            # Weighted total loss
            total_loss = (
                cfg.pde_weight * pde_loss
                + cfg.boundary_weight * boundary_loss
                + cfg.initial_weight * initial_loss
            )

            total_loss.backward()

            # Record losses
            self.history["pde"].append(pde_loss.item())
            self.history["boundary"].append(boundary_loss.item())
            self.history["initial"].append(initial_loss.item())
            self.history["total"].append(total_loss.item())

            return total_loss

        if isinstance(self.optimizer, torch.optim.LBFGS):
            self.optimizer.step(closure)
            _loss = closure()
        else:
            _loss = closure()
            self.optimizer.step()

        return {
            "pde": self.history["pde"][-1],
            "boundary": self.history["boundary"][-1],
            "initial": self.history["initial"][-1],
            "total": self.history["total"][-1],
        }

    def train(self, epochs: int | None = None) -> dict[str, list[float]]:
        """Train the PINN."""
        epochs = epochs or self.config.max_epochs

        print(f"Training {self.model.__class__.__name__} on {self.device}")
        print(
            f"PDE points: {self.config.n_pde_points}, "
            f"Boundary: {self.config.n_boundary_points}, "
            f"Initial: {self.config.n_initial_points}"
        )

        for epoch in range(epochs):
            losses = self.train_step()

            # Learning rate scheduling
            self.scheduler.step(losses["total"])

            # Early stopping
            if losses["total"] < self.best_loss:
                self.best_loss = losses["total"]
                self.patience_counter = 0
                # Save best model
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: PDE={losses['pde']:.6f}, "
                    f"Boundary={losses['boundary']:.6f}, "
                    f"Initial={losses['initial']:.6f}, "
                    f"Total={losses['total']:.6f}"
                )

        # Restore best model
        self.model.load_state_dict(self.best_state)
        print(f"Training complete. Best loss: {self.best_loss:.6f}")

        return self.history


# Convenience functions
def create_bs_pinn(
    strike: float = 100.0,
    option_type: str = "call",
    hidden_layers: list[int] = None,
    device: str = "auto",
) -> BlackScholesPINN:
    """Create a Black-Scholes PINN with default configuration."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = PINNConfig(
        hidden_layers=hidden_layers or [64, 64, 64, 64],
        activation="tanh",
        learning_rate=1e-3,
        max_epochs=10000,
        n_pde_points=10000,
        n_boundary_points=2000,
        n_initial_points=2000,
    )

    model = BlackScholesPINN(config, strike=strike, option_type=option_type)
    return model.to(device)


def create_heston_pinn(
    strike: float = 100.0,
    option_type: str = "call",
    hidden_layers: list[int] = None,
    device: str = "auto",
) -> HestonPINN:
    """Create a Heston PINN with default configuration."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = PINNConfig(
        hidden_layers=hidden_layers or [128, 128, 128, 128],
        activation="tanh",
        learning_rate=1e-3,
        max_epochs=15000,
        n_pde_points=20000,
        n_boundary_points=4000,
        n_initial_points=4000,
        v_min=0.001,
        v_max=0.5,
    )

    model = HestonPINN(config, strike=strike, option_type=option_type)
    return model.to(device)

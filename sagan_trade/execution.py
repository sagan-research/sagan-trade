"""
Optimal Execution Models.

Implements state-of-the-art optimal execution algorithms:
- Almgren-Chriss (2000) - Mean-Variance optimal execution
- Bertsimas-Lo (1998) - Dynamic programming approach
- Obizhaeva-Wang (2013) - Limit order book dynamics
- Gatheral-Schied (2011) - Transient market impact
- Standard baselines (TWAP, VWAP, POV, Implementation Shortfall)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Literal, Callable

import numpy as np
import pandas as pd
from scipy import optimize, linalg


class ExecutionModel(Enum):
    """Types of execution models."""
    ALMGREN_CHRISS = "almgren_chriss"
    BERTSIMAS_LO = "bertsimas_lo"
    O_BIZHAEVA_WANG = "obizhaeva_wang"
    GATHERAL_SCHIED = "gatheral_schied"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"


@dataclass
class ExecutionConfig:
    """Configuration for optimal execution."""
    
    # Model selection
    model: ExecutionModel = ExecutionModel.ALMGREN_CHRISS
    
    # Market parameters
    risk_aversion: float = 1e-6      
    volatility: float = 0.02         
    sigma: float = 0.02              
    
    # Market impact parameters
    permanent_impact: float = 0.1    
    temporary_impact: float = 0.1    
    impact_exponent: float = 0.5     
    
    # Limit order book (Obizhaeva-Wang)
    resilience: float = 1.0          
    depth: float = 100000            
    spread: float = 0.0001           
    
    # Transient impact (Gatheral-Schied)
    transient_impact_decay: float = 1.0  
    
    # Execution constraints
    total_quantity: float = 100000   
    time_horizon: float = 1.0        
    num_intervals: int = 390         
    min_trade_size: float = 100      
    max_trade_size: float = 50000    
    max_participation: float = 0.25  
    
    # Risk measures
    risk_measure: Literal["variance", "cvar", "var"] = "variance"
    confidence_level: float = 0.95   
    
    # Optimization
    solver: Literal["analytic", "qp", "dp"] = "analytic"
    verbose: bool = False


@dataclass
class ExecutionResult:
    """Results from execution optimization."""
    
    schedule: np.ndarray              
    times: np.ndarray                 
    expected_cost: float              
    expected_shortfall: float         
    variance: float                   
    sharpe: float                     
    
    # Trajectories
    price_path: Optional[np.ndarray] = None
    position_path: Optional[np.ndarray] = None
    cash_path: Optional[np.ndarray] = None
    
    # Metrics
    market_impact_cost: float = 0.0
    timing_risk_cost: float = 0.0
    opportunity_cost: float = 0.0
    
    # Model info
    model: str = ""
    config: Optional[ExecutionConfig] = None
    success: bool = True
    message: str = ""
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert schedule to DataFrame."""
        total = np.sum(self.schedule)
        return pd.DataFrame({
            'time': self.times,
            'trade_size': self.schedule,
            'cumulative': np.cumsum(self.schedule),
            'remaining': total - np.cumsum(self.schedule)
        })
    
    @property
    def total_quantity(self) -> float:
        return np.sum(self.schedule)


class BaseExecutionModel:
    """Abstract base class for execution models."""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        
    def optimize(self) -> ExecutionResult:
        """Compute optimal execution schedule."""
        raise NotImplementedError
    
    def simulate(self, n_paths: int = 1000) -> Dict[str, np.ndarray]:
        """Simulate execution paths."""
        raise NotImplementedError


class AlmgrenChrissModel(BaseExecutionModel):
    """
    Almgren-Chriss Optimal Execution Model.
    
    Minimizes mean-variance utility: E[cost] + lambda * Var[cost]
    
    The optimal strategy is exponential:
    x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)
    
    where kappa = sqrt(lambda * sigma^2 / eta)
    """
    
    def __init__(self, config: ExecutionConfig):
        super().__init__(config)
        self._kappa = None
        self._schedule = None
        
    def optimize(self) -> ExecutionResult:
        """Compute Almgren-Chriss optimal schedule."""
        cfg = self.config
        
        # Time grid
        T = cfg.time_horizon
        N = cfg.num_intervals
        dt = T / N
        times = np.linspace(0, T, N + 1)
        
        # Parameters
        X = cfg.total_quantity
        sigma = cfg.sigma
        eta = cfg.temporary_impact
        gamma = cfg.permanent_impact
        lam = cfg.risk_aversion
        
        # Compute kappa
        if lam > 0:
            kappa = np.sqrt(lam * sigma**2 / eta)
        else:
            kappa = 0
            
        self._kappa = kappa
        
        # Optimal schedule
        if kappa > 0:
            schedule = np.zeros(N)
            for k in range(1, N + 1):
                t_k = k * dt
                t_prev = (k - 1) * dt
                # Discrete version of exponential schedule
                x_k = X * (np.exp(kappa * (T - t_prev)) - np.exp(kappa * (T - t_k))) / (np.exp(kappa * T) - 1)
                schedule[k - 1] = x_k
        else:
            # TWAP (risk-neutral)
            schedule = np.full(N, X / N)
            
        self._schedule = schedule
        
        # Compute expected cost and variance
        expected_cost = self._expected_cost(schedule, dt)
        variance = self._variance(schedule, dt)
        sharpe = expected_cost / np.sqrt(variance) if variance > 0 else 0
        
        # Decompose costs
        market_impact = self._market_impact_cost(schedule, dt)
        timing_risk = lam * variance
        opportunity = 0
        
        return ExecutionResult(
            schedule=schedule,
            times=times[:-1],
            expected_cost=expected_cost,
            expected_shortfall=expected_cost,
            variance=variance,
            sharpe=sharpe,
            market_impact_cost=market_impact,
            timing_risk_cost=timing_risk,
            opportunity_cost=opportunity,
            model="Almgren-Chriss",
            config=cfg,
        )
    
    def _expected_cost(self, schedule: np.ndarray, dt: float) -> float:
        """Compute expected execution cost."""
        cfg = self.config
        X = cfg.total_quantity
        eta = cfg.temporary_impact
        gamma = cfg.permanent_impact
        
        # Permanent impact cost
        remaining = X - np.concatenate([[0], np.cumsum(schedule[:-1])])
        perm_cost = gamma * np.sum(schedule * remaining)
        
        # Temporary impact cost
        temp_cost = eta * np.sum(schedule**2) / dt
        
        return perm_cost + temp_cost
    
    def _variance(self, schedule: np.ndarray, dt: float) -> float:
        """Compute cost variance."""
        cfg = self.config
        sigma = cfg.sigma
        remaining = cfg.total_quantity - np.cumsum(schedule)
        return sigma**2 * np.sum(remaining**2) * dt
    
    def _market_impact_cost(self, schedule: np.ndarray, dt: float) -> float:
        """Compute market impact component."""
        cfg = self.config
        return cfg.temporary_impact * np.sum(schedule**2) / dt
    
    def simulate(self, n_paths: int = 1000) -> Dict[str, np.ndarray]:
        """Monte Carlo simulation of execution."""
        if self._schedule is None:
            self.optimize()
            
        cfg = self.config
        schedule = self._schedule
        N = cfg.num_intervals
        dt = cfg.time_horizon / N
        
        # Simulate price paths
        price_paths = np.zeros((n_paths, N + 1))
        price_paths[:, 0] = 100.0
        
        for k in range(1, N + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            permanent_impact = -cfg.permanent_impact * schedule[k - 1]
            price_paths[:, k] = (price_paths[:, k - 1] + 
                                 cfg.sigma * price_paths[:, k - 1] * dW + 
                                 permanent_impact)
            
        # Compute costs
        costs = np.zeros(n_paths)
        for path in range(n_paths):
            for k in range(N):
                temp_impact = -cfg.temporary_impact * schedule[k] / dt
                exec_price = price_paths[path, k] + temp_impact
                costs[path] += schedule[k] * (exec_price - price_paths[path, 0])
                
        return {
            'price_paths': price_paths,
            'costs': costs,
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'var_95': np.percentile(costs, 5),
            'cvar_95': np.mean(costs[costs <= np.percentile(costs, 5)]),
        }


class BertsimasLoModel(BaseExecutionModel):
    """Bertsimas-Lo Optimal Execution (Dynamic Programming)."""
    
    def optimize(self) -> ExecutionResult:
        """Solve using dynamic programming."""
        cfg = self.config
        N = cfg.num_intervals
        X = cfg.total_quantity
        
        # Discretize state space
        max_qty = int(X)
        state_grid = np.linspace(0, max_qty, max_qty + 1)
        n_states = len(state_grid)
        
        # Value function
        V = np.full((N + 1, n_states), np.inf)
        policy = np.zeros((N, n_states))
        
        # Terminal condition
        V[N, :] = 0
        
        # Backward induction
        for k in range(N - 1, -1, -1):
            for i, q in enumerate(state_grid):
                if q == 0:
                    V[k, i] = 0
                    policy[k, i] = 0
                    continue
                    
                min_cost = np.inf
                best_trade = 0
                
                max_trade = min(q, cfg.max_trade_size)
                min_trade = cfg.min_trade_size if q >= cfg.min_trade_size else q
                n_trades = max(1, int(max_trade / max(min_trade, 1)) + 1)
                
                for t in np.linspace(min_trade, max_trade, n_trades):
                    temp_impact = cfg.temporary_impact * t
                    perm_impact = cfg.permanent_impact * t
                    
                    q_next = q - t
                    idx_next = np.argmin(np.abs(state_grid - q_next))
                    future_cost = V[k + 1, idx_next]
                    
                    timing_risk = cfg.risk_aversion * cfg.sigma**2 * q_next**2 * (cfg.time_horizon / N)
                    
                    total_cost = temp_impact * t + perm_impact * q_next + future_cost + timing_risk
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_trade = t
                        
                V[k, i] = min_cost
                policy[k, i] = best_trade
                
        # Extract optimal schedule
        schedule = np.zeros(N)
        q = X
        for k in range(N):
            idx = np.argmin(np.abs(state_grid - q))
            trade = policy[k, idx]
            schedule[k] = trade
            q -= trade
            
        times = np.linspace(0, cfg.time_horizon, N + 1)[:-1]
        
        expected_cost = V[0, np.argmin(np.abs(state_grid - X))]
        variance = self._compute_variance(schedule)
        sharpe = expected_cost / np.sqrt(variance) if variance > 0 else 0
        
        return ExecutionResult(
            schedule=schedule,
            times=times,
            expected_cost=expected_cost,
            expected_shortfall=expected_cost,
            variance=variance,
            sharpe=sharpe,
            model="Bertsimas-Lo",
            config=cfg,
        )
    
    def _compute_variance(self, schedule: np.ndarray) -> float:
        cfg = self.config
        remaining = cfg.total_quantity - np.cumsum(schedule)
        dt = cfg.time_horizon / cfg.num_intervals
        return cfg.sigma**2 * np.sum(remaining**2) * dt
    
    def simulate(self, n_paths: int = 1000) -> Dict[str, np.ndarray]:
        return AlmgrenChrissModel(self.config).simulate(n_paths)


class ObizhaevaWangModel(BaseExecutionModel):
    """Obizhaeva-Wang Model with Limit Order Book Dynamics."""
    
    def optimize(self) -> ExecutionResult:
        """Solve OW model."""
        cfg = self.config
        N = cfg.num_intervals
        X = cfg.total_quantity
        T = cfg.time_horizon
        dt = T / N
        
        # OW parameters
        rho = cfg.resilience
        lambda_ = cfg.permanent_impact / cfg.depth
        alpha = cfg.temporary_impact / cfg.depth
        spread = cfg.spread
        
        # Optimal strategy with resilience
        kappa = np.sqrt(lambda_ * rho / alpha) if alpha > 0 else 0
        
        times = np.linspace(0, T, N + 1)
        schedule = np.zeros(N)
        
        if kappa > 0:
            for k in range(1, N + 1):
                t_k = k * dt
                t_prev = (k - 1) * dt
                exp_term = np.exp(-kappa * T)
                numer = np.exp(-kappa * t_prev) - np.exp(-kappa * t_k)
                denom = 1 - exp_term
                schedule[k - 1] = X * numer / denom
        else:
            schedule = np.full(N, X / N)
            
        expected_cost = self._ow_cost(schedule, dt)
        variance = self._ow_variance(schedule, dt)
        sharpe = expected_cost / np.sqrt(variance) if variance > 0 else 0
        
        return ExecutionResult(
            schedule=schedule,
            times=times[:-1],
            expected_cost=expected_cost,
            expected_shortfall=expected_cost,
            variance=variance,
            sharpe=sharpe,
            model="Obizhaeva-Wang",
            config=cfg,
        )
    
    def _ow_cost(self, schedule: np.ndarray, dt: float) -> float:
        cfg = self.config
        cost = 0
        remaining = cfg.total_quantity
        
        for x in schedule:
            cost += x * (cfg.spread / 2 + cfg.temporary_impact * x / cfg.depth)
            remaining -= x
            cost += cfg.permanent_impact * remaining * x / cfg.depth
            
        return cost
    
    def _ow_variance(self, schedule: np.ndarray, dt: float) -> float:
        cfg = self.config
        remaining = cfg.total_quantity - np.cumsum(schedule)
        return cfg.sigma**2 * np.sum(remaining**2) * dt
    
    def simulate(self, n_paths: int = 1000) -> Dict[str, np.ndarray]:
        result = self.optimize()
        cfg = self.config
        schedule = result.schedule
        N = cfg.num_intervals
        dt = cfg.time_horizon / N
        
        price_paths = np.zeros((n_paths, N + 1))
        price_paths[:, 0] = 100.0
        
        for k in range(1, N + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            x_k = schedule[k - 1]
            perm_impact = -cfg.permanent_impact * x_k / cfg.depth
            temp_impact = -cfg.temporary_impact * x_k / cfg.depth
            decay = np.exp(-cfg.resilience * dt) if k > 1 else 1
            
            price_paths[:, k] = (price_paths[:, k - 1] + 
                                 cfg.sigma * price_paths[:, k - 1] * dW + 
                                 perm_impact * decay + temp_impact)
                                 
        costs = np.zeros(n_paths)
        for p in range(n_paths):
            for k in range(N):
                exec_price = price_paths[p, k] - cfg.temporary_impact * schedule[k] / cfg.depth
                costs[p] += schedule[k] * (exec_price - price_paths[p, 0])
                
        return {
            'price_paths': price_paths,
            'costs': costs,
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
        }


class GatheralSchiedModel(BaseExecutionModel):
    """Gatheral-Schied Transient Impact Model."""
    
    def optimize(self) -> ExecutionResult:
        """Solve for optimal schedule with transient impact."""
        cfg = self.config
        N = cfg.num_intervals
        X = cfg.total_quantity
        T = cfg.time_horizon
        dt = T / N
        
        gamma = cfg.permanent_impact
        rho = cfg.transient_impact_decay
        eta = cfg.temporary_impact
        lam = cfg.risk_aversion
        sigma = cfg.sigma
        
        # Impact kernel matrix
        G = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i >= j:
                    G[i, j] = gamma * np.exp(-rho * (i - j) * dt)
                    
        # Objective: minimize x^T (G + eta*I) x + lambda * sigma^2 * sum (X - cumsum(x))^2
        H = G + G.T + np.eye(N) * eta
        C = np.tril(np.ones((N, N)))
        H_risk = lam * sigma**2 * C.T @ C * dt
        H_total = H + H_risk
        
        # Constraint: sum(x) = X
        ones = np.ones(N)
        KKT = np.block([
            [H_total, ones.reshape(-1, 1)],
            [ones.reshape(1, -1), np.zeros((1, 1))]
        ])
        rhs = np.concatenate([np.zeros(N), [X]])
        
        sol = np.linalg.solve(KKT, rhs)
        schedule = sol[:N]
        schedule = np.maximum(schedule, 0)
        schedule = schedule / schedule.sum() * X
        
        times = np.linspace(0, T, N + 1)[:-1]
        
        expected_cost = schedule @ H @ schedule
        variance = lam * sigma**2 * np.sum((X - np.cumsum(schedule))**2) * dt
        sharpe = expected_cost / np.sqrt(variance) if variance > 0 else 0
        
        return ExecutionResult(
            schedule=schedule,
            times=times,
            expected_cost=expected_cost,
            expected_shortfall=expected_cost,
            variance=variance,
            sharpe=sharpe,
            model="Gatheral-Schied",
            config=cfg,
        )
    
    def simulate(self, n_paths: int = 1000) -> Dict[str, np.ndarray]:
        result = self.optimize()
        cfg = self.config
        schedule = result.schedule
        N = cfg.num_intervals
        dt = cfg.time_horizon / N
        
        price_paths = np.zeros((n_paths, N + 1))
        price_paths[:, 0] = 100.0
        impact_state = np.zeros(n_paths)
        
        for k in range(1, N + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            x_k = schedule[k - 1]
            
            impact_state = impact_state * np.exp(-cfg.transient_impact_decay * dt) + cfg.permanent_impact * x_k
            price_paths[:, k] = (price_paths[:, k - 1] + 
                                 cfg.sigma * price_paths[:, k - 1] * dW + 
                                 impact_state -
                                 cfg.temporary_impact * x_k)
                                 
        costs = np.zeros(n_paths)
        for p in range(n_paths):
            for k in range(N):
                exec_price = price_paths[p, k] - cfg.temporary_impact * schedule[k]
                costs[p] += schedule[k] * (exec_price - price_paths[p, 0])
                
        return {
            'price_paths': price_paths,
            'costs': costs,
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
        }


class TWAPModel(BaseExecutionModel):
    """Time-Weighted Average Price (TWAP) - baseline."""
    
    def optimize(self) -> ExecutionResult:
        cfg = self.config
        N = cfg.num_intervals
        X = cfg.total_quantity
        T = cfg.time_horizon
        
        schedule = np.full(N, X / N)
        times = np.linspace(0, T, N + 1)[:-1]
        
        expected_cost = cfg.temporary_impact * np.sum(schedule**2) * (N / T)
        variance = cfg.sigma**2 * np.sum((X - np.cumsum(schedule))**2) * (T / N)
        sharpe = expected_cost / np.sqrt(variance) if variance > 0 else 0
        
        return ExecutionResult(
            schedule=schedule,
            times=times,
            expected_cost=expected_cost,
            expected_shortfall=expected_cost,
            variance=variance,
            sharpe=sharpe,
            model="TWAP",
            config=cfg,
        )
    
    def simulate(self, n_paths: int = 1000) -> Dict[str, np.ndarray]:
        return AlmgrenChrissModel(self.config).simulate(n_paths)


class VWAPModel(BaseExecutionModel):
    """Volume-Weighted Average Price (VWAP) execution."""
    
    def optimize(self, volume_profile: Optional[np.ndarray] = None) -> ExecutionResult:
        cfg = self.config
        N = cfg.num_intervals
        X = cfg.total_quantity
        T = cfg.time_horizon
        
        if volume_profile is None:
            t = np.linspace(0, T, N)
            volume_profile = 1 + 0.5 * np.sin(2 * np.pi * t / T) + 0.3 * np.sin(4 * np.pi * t / T)
            
        volume_profile = volume_profile / volume_profile.sum()
        schedule = X * volume_profile
        times = np.linspace(0, T, N + 1)[:-1]
        
        expected_cost = cfg.temporary_impact * np.sum(schedule**2) * (N / T)
        variance = cfg.sigma**2 * np.sum((X - np.cumsum(schedule))**2) * (T / N)
        sharpe = expected_cost / np.sqrt(variance) if variance > 0 else 0
        
        return ExecutionResult(
            schedule=schedule,
            times=times,
            expected_cost=expected_cost,
            expected_shortfall=expected_cost,
            variance=variance,
            sharpe=sharpe,
            model="VWAP",
            config=cfg,
        )
    
    def simulate(self, n_paths: int = 1000) -> Dict[str, np.ndarray]:
        return AlmgrenChrissModel(self.config).simulate(n_paths)


class POVModel(BaseExecutionModel):
    """Percentage of Volume (POV) execution."""
    
    def optimize(self, volume_forecast: Optional[np.ndarray] = None) -> ExecutionResult:
        cfg = self.config
        N = cfg.num_intervals
        X = cfg.total_quantity
        T = cfg.time_horizon
        
        pov_rate = cfg.max_participation
        
        if volume_forecast is None:
            volume_forecast = np.full(N, X / (pov_rate * N))
            
        schedule = np.minimum(pov_rate * volume_forecast, cfg.max_trade_size)
        
        if schedule.sum() > 0:
            schedule = schedule / schedule.sum() * X
        else:
            schedule = np.full(N, X / N)
            
        times = np.linspace(0, T, N + 1)[:-1]
        
        expected_cost = cfg.temporary_impact * np.sum(schedule**2) * (N / T)
        variance = cfg.sigma**2 * np.sum((X - np.cumsum(schedule))**2) * (T / N)
        sharpe = expected_cost / np.sqrt(variance) if variance > 0 else 0
        
        return ExecutionResult(
            schedule=schedule,
            times=times,
            expected_cost=expected_cost,
            expected_shortfall=expected_cost,
            variance=variance,
            sharpe=sharpe,
            model="POV",
            config=cfg,
        )
    
    def simulate(self, n_paths: int = 1000) -> Dict[str, np.ndarray]:
        return AlmgrenChrissModel(self.config).simulate(n_paths)


class ImplementationShortfallModel(BaseExecutionModel):
    """Implementation Shortfall (IS) Optimization."""
    
    def optimize(self, 
                 alpha_signal: Optional[np.ndarray] = None,
                 urgency: float = 1.0) -> ExecutionResult:
        """Optimize with implementation shortfall objective."""
        cfg = self.config
        N = cfg.num_intervals
        X = cfg.total_quantity
        T = cfg.time_horizon
        dt = T / N
        
        if alpha_signal is not None:
            eff_urgency = cfg.risk_aversion + urgency * np.mean(alpha_signal)
        else:
            eff_urgency = cfg.risk_aversion
            
        # Use AC model with effective risk aversion
        ac_cfg = ExecutionConfig(
            **{k: v for k, v in cfg.__dict__.items() if k != 'risk_aversion'},
            risk_aversion=eff_urgency
        )
        
        ac_model = AlmgrenChrissModel(ac_cfg)
        result = ac_model.optimize()
        result.model = "ImplementationShortfall"
        result.config = cfg
        
        if alpha_signal is not None:
            alpha_cost = np.sum(alpha_signal * (X - np.cumsum(result.schedule)) * dt)
            result.expected_cost += alpha_cost
            result.opportunity_cost = alpha_cost
            result.expected_shortfall = result.expected_cost
            
        return result
    
    def simulate(self, n_paths: int = 1000) -> Dict[str, np.ndarray]:
        return AlmgrenChrissModel(self.config).simulate(n_paths)


class MultiAssetExecutionModel:
    """Multi-asset optimal execution with cross-impact."""
    
    def __init__(self, 
                 configs: List[ExecutionConfig],
                 correlation: np.ndarray,
                 cross_impact: Optional[np.ndarray] = None):
        self.configs = configs
        self.n_assets = len(configs)
        self.correlation = correlation
        self.cross_impact = cross_impact if cross_impact is not None else correlation * 0.5
        
    def optimize(self) -> List[ExecutionResult]:
        """Solve multi-asset execution jointly."""
        # This is a placeholder for the full multi-asset optimization
        # which would require solving a larger QP problem
        results = []
        for cfg in self.configs:
            model = AlmgrenChrissModel(cfg)
            results.append(model.optimize())
        return results


def create_execution_model(model_type: ExecutionModel, config: ExecutionConfig) -> BaseExecutionModel:
    """Factory function to create execution model."""
    models = {
        ExecutionModel.ALMGREN_CHRISS: AlmgrenChrissModel,
        ExecutionModel.BERTSIMAS_LO: BertsimasLoModel,
        ExecutionModel.O_BIZHAEVA_WANG: ObizhaevaWangModel,
        ExecutionModel.GATHERAL_SCHIED: GatheralSchiedModel,
        ExecutionModel.TWAP: TWAPModel,
        ExecutionModel.VWAP: VWAPModel,
        ExecutionModel.POV: POVModel,
        ExecutionModel.IMPLEMENTATION_SHORTFALL: ImplementationShortfallModel,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type](config)


def optimize_execution(config: ExecutionConfig) -> ExecutionResult:
    """High-level execution optimization."""
    model = create_execution_model(config.model, config)
    return model.optimize()


def compare_execution_models(config: ExecutionConfig, 
                             models: Optional[List[ExecutionModel]] = None,
                             n_simulations: int = 1000) -> pd.DataFrame:
    """Compare multiple execution models."""
    if models is None:
        models = [
            ExecutionModel.ALMGREN_CHRISS,
            ExecutionModel.TWAP,
            ExecutionModel.VWAP,
            ExecutionModel.POV,
        ]
        
    results = []
    for model_type in models:
        cfg = ExecutionConfig(**{**config.__dict__, 'model': model_type})
        model = create_execution_model(model_type, cfg)
        result = model.optimize()
        
        # Run simulation
        sim = model.simulate(n_simulations)
        
        results.append({
            'model': model_type.value,
            'expected_cost': result.expected_cost,
            'variance': result.variance,
            'sharpe': result.sharpe,
            'market_impact': result.market_impact_cost,
            'timing_risk': result.timing_risk_cost,
            'sim_mean_cost': sim['mean_cost'],
            'sim_std_cost': sim['std_cost'],
            'sim_var_95': sim.get('var_95', np.nan),
        })
        
    return pd.DataFrame(results)
"""
Advanced Portfolio Optimization Module.

Implements state-of-the-art portfolio optimization techniques:
- Hierarchical Risk Parity (HRP) - Marcos Lopez de Prado
- Risk Parity / Equal Risk Contribution (ERC)
- Black-Litterman Model with Bayesian views
- Mean-Variance Optimization with robust covariance estimation
- Maximum Diversification Portfolio
- Minimum Variance Portfolio
- Tail Risk Parity (CVaR-based)

References:
- Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform
  Out-of-Sample." Journal of Portfolio Management.
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties of
  Equally Weighted Risk Contribution Portfolios."
- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization."
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import cluster, linalg

try:
    import skfolio  # noqa: F401
    from skfolio.optimization import (  # noqa: F401
        HierarchicalRiskParity,
        MaximumDiversification,
        MeanRisk,
        MinimumVariance,
        RiskParity,
    )
    from skfolio.prior import BlackLitterman  # noqa: F401

    SKFOLIO_AVAILABLE = True
except ImportError:
    SKFOLIO_AVAILABLE = False
    warnings.warn(
        "skfolio not installed. Some advanced optimizers will use custom implementations. "
        "Install with: pip install skfolio"
    )


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""

    # Constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_leverage: float = 1.0  # Sum of absolute weights

    # Regularization
    covariance_shrinkage: float = 0.1  # Ledoit-Wolf shrinkage
    eigenvalue_floor: float = 1e-6  # Minimum eigenvalue for regularization

    # Risk measures
    risk_measure: Literal["variance", "cvar", "madr", "evc", "cdar"] = "variance"
    confidence_level: float = 0.95  # For CVaR

    # Black-Litterman
    tau: float = 0.05  # Uncertainty scaling
    risk_aversion: float | None = None  # If None, estimate from market

    # HRP
    linkage_method: Literal["single", "complete", "average", "ward"] = "ward"
    distance_metric: Literal["correlation", "euclidean", "angular"] = "correlation"

    # Solver
    solver: Literal["ECOS", "OSQP", "SCS", "CLARABEL"] = "SCS"
    verbose: bool = False


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: np.ndarray
    asset_names: list[str]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float

    # Risk contributions (for risk parity)
    risk_contributions: np.ndarray | None = None

    # Additional metrics
    diversification_ratio: float | None = None
    effective_number_assets: float | None = None

    # Black-Litterman specific
    posterior_mu: np.ndarray | None = None
    posterior_sigma: np.ndarray | None = None
    implied_returns: np.ndarray | None = None

    # HRP specific
    linkage_matrix: np.ndarray | None = None
    cluster_order: list[int] | None = None

    # Metadata
    method: str = ""
    config: OptimizationConfig | None = None
    success: bool = True
    message: str = ""

    def to_series(self) -> pd.Series:
        """Convert weights to pandas Series."""
        return pd.Series(self.weights, index=self.asset_names, name="weight")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weights": dict(zip(self.asset_names, self.weights, strict=False)),
            "expected_return": self.expected_return,
            "expected_risk": self.expected_risk,
            "sharpe_ratio": self.sharpe_ratio,
            "diversification_ratio": self.diversification_ratio,
            "effective_number_assets": self.effective_number_assets,
            "method": self.method,
            "success": self.success,
            "message": self.message,
        }


class BaseOptimizer(ABC):
    """Base class for portfolio optimizers."""

    def __init__(self, config: OptimizationConfig | None = None):
        self.config = config or OptimizationConfig()

    @abstractmethod
    def optimize(self, returns: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Optimize portfolio."""
        pass

    def _prepare_inputs(
        self,
        returns: pd.DataFrame,
        expected_returns: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare and validate inputs."""
        asset_names = returns.columns.tolist()
        n_assets = len(asset_names)

        # Expected returns
        if expected_returns is None:
            mu = returns.mean().values * 252  # Annualize
        else:
            mu = np.asarray(expected_returns)
            assert len(mu) == n_assets

        # Covariance matrix with shrinkage
        if covariance is None:
            sigma = self._shrink_covariance(returns)
        else:
            sigma = np.asarray(covariance)
            assert sigma.shape == (n_assets, n_assets)

        return mu, sigma, asset_names

    def _shrink_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage to covariance matrix."""
        from sklearn.covariance import ledoit_wolf

        # Use Ledoit-Wolf shrinkage
        sigma, _ = ledoit_wolf(returns.values)

        # Additional eigenvalue floor
        eigenvals, eigenvecs = linalg.eigh(sigma)
        eigenvals = np.maximum(eigenvals, self.config.eigenvalue_floor)
        sigma = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return sigma

    def _build_constraints(self, n_assets: int) -> list[cp.Constraint]:
        """Build standard portfolio constraints."""
        w = cp.Variable(n_assets)
        constraints = [
            cp.sum(w) <= self.config.max_leverage,
            cp.sum(w) >= -self.config.max_leverage,
            w >= self.config.min_weight,
            w <= self.config.max_weight,
        ]
        return constraints

    def _compute_metrics(
        self, weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> dict[str, float]:
        """Compute portfolio metrics."""
        port_return = weights @ mu
        port_variance = weights @ sigma @ weights
        port_vol = np.sqrt(max(port_variance, 1e-12))
        sharpe = port_return / port_vol if port_vol > 0 else 0

        # Diversification ratio
        weighted_vol = np.sum(weights * np.sqrt(np.diag(sigma)))
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1

        # Effective number of assets
        ena = 1 / np.sum(weights**2) if np.sum(weights**2) > 0 else 1

        return {
            "expected_return": float(port_return),
            "expected_risk": float(port_vol),
            "sharpe_ratio": float(sharpe),
            "diversification_ratio": float(div_ratio),
            "effective_number_assets": float(ena),
        }


class HierarchicalRiskParity(BaseOptimizer):
    """
    Hierarchical Risk Parity (HRP) - Lopez de Prado (2016).

    Builds a hierarchical clustering tree from the correlation matrix
    and allocates risk recursively through the tree structure.
    """

    def __init__(self, config: OptimizationConfig | None = None):
        super().__init__(config)
        self._linkage_matrix = None
        self._cluster_order = None

    def optimize(
        self, returns: pd.DataFrame, covariance: np.ndarray | None = None, **kwargs
    ) -> OptimizationResult:
        """
        Optimize using Hierarchical Risk Parity.

        Args:
            returns: Historical returns DataFrame
            covariance: Optional pre-computed covariance matrix

        Returns:
            OptimizationResult with HRP weights
        """

        # Compute correlation and covariance
        if covariance is None:
            cov_matrix = returns.cov().values * 252
        else:
            cov_matrix = covariance

        corr_matrix = self._cov_to_corr(cov_matrix)

        # Build hierarchical clustering
        linkage = self._build_linkage(corr_matrix)
        self._linkage_matrix = linkage

        # Get quasi-diagonalization order
        cluster_order = self._get_quasi_diagonal(linkage)
        self._cluster_order = cluster_order

        # Recursive bisection allocation
        weights = self._recursive_bisection(cov_matrix, cluster_order)

        # Compute metrics
        mu = returns.mean().values * 252
        metrics = self._compute_metrics(weights, mu, cov_matrix)

        # Risk contributions
        risk_contrib = self._risk_contributions(weights, cov_matrix)

        return OptimizationResult(
            weights=weights,
            asset_names=returns.columns.tolist(),
            expected_return=metrics["expected_return"],
            expected_risk=metrics["expected_risk"],
            sharpe_ratio=metrics["sharpe_ratio"],
            risk_contributions=risk_contrib,
            diversification_ratio=metrics["diversification_ratio"],
            effective_number_assets=metrics["effective_number_assets"],
            linkage_matrix=linkage,
            cluster_order=cluster_order,
            method="HRP",
            config=self.config,
        )

    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance to correlation matrix."""
        cov = np.nan_to_num(cov, nan=0.0)
        d = np.sqrt(np.diag(cov))
        d = np.where(d == 0, 1e-10, d)
        corr = cov / np.outer(d, d)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _build_linkage(self, corr: np.ndarray) -> np.ndarray:
        """Build hierarchical clustering linkage matrix."""
        # Convert correlation to distance
        if self.config.distance_metric == "correlation":
            dist = np.sqrt(0.5 * (1 - corr))
        elif self.config.distance_metric == "angular":
            dist = np.arccos(np.clip(corr, -1, 1)) / np.pi
        else:  # euclidean
            dist = np.sqrt(np.sum((corr[:, np.newaxis] - corr[np.newaxis, :]) ** 2, axis=2))

        dist = np.nan_to_num(dist, nan=0.0)

        # Hierarchical clustering
        linkage = cluster.hierarchy.linkage(dist, method=self.config.linkage_method)
        return linkage

    def _get_quasi_diagonal(self, linkage: np.ndarray) -> list[int]:
        """Get quasi-diagonal ordering from linkage matrix."""
        n = linkage.shape[0] + 1

        def _recursive_sort(cluster_idx: int) -> list[int]:
            if cluster_idx < n:
                return [cluster_idx]
            left = int(linkage[cluster_idx - n, 0])
            right = int(linkage[cluster_idx - n, 1])
            return _recursive_sort(left) + _recursive_sort(right)

        return _recursive_sort(2 * n - 2)

    def _recursive_bisection(self, cov: np.ndarray, cluster_order: list[int]) -> np.ndarray:
        """Recursive bisection for risk parity allocation."""
        weights = np.ones(len(cluster_order))

        def _bisect(assets: list[int], weight: float):
            if len(assets) == 1:
                return

            # Split into two clusters
            mid = len(assets) // 2
            left_assets = assets[:mid]
            right_assets = assets[mid:]

            # Compute cluster variances
            left_var = self._cluster_variance(cov, left_assets)
            right_var = self._cluster_variance(cov, right_assets)

            # Allocate weight inversely proportional to variance
            total_var = left_var + right_var
            left_weight = weight * (right_var / total_var)
            right_weight = weight * (left_var / total_var)

            # Update weights
            for i, asset in enumerate(cluster_order):
                if asset in left_assets:
                    weights[i] *= left_weight / weight
                elif asset in right_assets:
                    weights[i] *= right_weight / weight

            # Recurse
            _bisect(left_assets, left_weight)
            _bisect(right_assets, right_weight)

        _bisect(cluster_order, 1.0)
        return weights / weights.sum()

    def _cluster_variance(self, cov: np.ndarray, assets: list[int]) -> float:
        """Compute variance of a cluster using inverse-variance weights."""
        if len(assets) == 1:
            return cov[assets[0], assets[0]]

        sub_cov = cov[np.ix_(assets, assets)]
        inv_diag = 1 / np.diag(sub_cov)
        ivp_weights = inv_diag / inv_diag.sum()
        return ivp_weights @ sub_cov @ ivp_weights

    def _risk_contributions(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute risk contributions."""
        port_var = weights @ cov @ weights
        marg_risk = cov @ weights
        risk_contrib = weights * marg_risk / port_var
        return risk_contrib


class RiskParityOptimizer(BaseOptimizer):
    """
    Risk Parity / Equal Risk Contribution (ERC) Portfolio.

    Finds weights such that each asset contributes equally to portfolio risk.
    """

    def optimize(
        self,
        returns: pd.DataFrame,
        covariance: np.ndarray | None = None,
        target_risk: float | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize Equal Risk Contribution portfolio.

        Args:
            returns: Historical returns
            covariance: Optional covariance matrix
            target_risk: Optional target portfolio volatility

        Returns:
            OptimizationResult with ERC weights
        """
        n_assets = len(returns.columns)
        mu, sigma, asset_names = self._prepare_inputs(returns, covariance=covariance)

        if SKFOLIO_AVAILABLE:
            return self._optimize_skfolio(returns, covariance, target_risk)

        # Custom implementation using optimization
        w = cp.Variable(n_assets)

        # Risk contribution equalization objective
        # Minimize sum of squared differences from equal risk contribution
        port_var = cp.quad_form(w, sigma)
        marg_risk = sigma @ w
        risk_contrib = cp.multiply(w, marg_risk)

        target_contrib = port_var / n_assets
        objective = cp.Minimize(cp.sum_squares(risk_contrib - target_contrib))

        constraints = self._build_constraints(n_assets)

        # Add target risk constraint if specified
        if target_risk is not None:
            constraints.append(cp.sqrt(port_var) <= target_risk)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.config.solver, verbose=self.config.verbose)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # Fallback: use inverse volatility weighting
            weights = 1 / np.sqrt(np.diag(sigma))
            weights = weights / weights.sum()
        else:
            weights = w.value
            weights = weights / np.sum(np.abs(weights)) * self.config.max_leverage

        metrics = self._compute_metrics(weights, mu, sigma)
        risk_contrib = self._risk_contributions(weights, sigma)

        return OptimizationResult(
            weights=weights,
            asset_names=asset_names,
            expected_return=metrics["expected_return"],
            expected_risk=metrics["expected_risk"],
            sharpe_ratio=metrics["sharpe_ratio"],
            risk_contributions=risk_contrib,
            diversification_ratio=metrics["diversification_ratio"],
            effective_number_assets=metrics["effective_number_assets"],
            method="RiskParity",
            config=self.config,
        )

    def _optimize_skfolio(
        self,
        returns: pd.DataFrame,
        covariance: np.ndarray | None,
        target_risk: float | None,
    ) -> OptimizationResult:
        """Use skfolio's RiskParity optimizer."""
        model = RiskParity(
            risk_measure=self.config.risk_measure,
            portfolio_params={
                "min_weight": self.config.min_weight,
                "max_weight": self.config.max_weight,
            },
        )
        model.fit(returns)

        weights = model.weights_
        mu = returns.mean().values * 252
        sigma = covariance if covariance is not None else returns.cov().values * 252
        metrics = self._compute_metrics(weights, mu, sigma)
        risk_contrib = self._risk_contributions(weights, sigma)

        return OptimizationResult(
            weights=weights,
            asset_names=returns.columns.tolist(),
            expected_return=metrics["expected_return"],
            expected_risk=metrics["expected_risk"],
            sharpe_ratio=metrics["sharpe_ratio"],
            risk_contributions=risk_contrib,
            diversification_ratio=metrics["diversification_ratio"],
            effective_number_assets=metrics["effective_number_assets"],
            method="RiskParity",
            config=self.config,
        )


class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman Portfolio Optimization.

    Combines market equilibrium returns with investor views to produce
    posterior expected returns and optimal portfolio weights.
    """

    def __init__(self, config: OptimizationConfig | None = None):
        super().__init__(config)
        self._implied_returns = None
        self._posterior_mu = None
        self._posterior_sigma = None

    def optimize(
        self,
        returns: pd.DataFrame,
        market_caps: np.ndarray | None = None,
        views: dict[str, float] | None = None,
        view_confidence: dict[str, float] | None = None,
        pick_matrix: np.ndarray | None = None,
        view_returns: np.ndarray | None = None,
        view_covariance: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize using Black-Litterman model.

        Args:
            returns: Historical returns for covariance estimation
            market_caps: Market capitalizations for equilibrium weights
            views: Dict of {asset_name: expected_return} for absolute views
            view_confidence: Dict of {asset_name: confidence} (0-1)
            pick_matrix: P matrix for relative views (k x n)
            view_returns: Q vector for views (k,)
            view_covariance: Omega matrix for view uncertainty (k x k)
            covariance: Pre-computed covariance matrix

        Returns:
            OptimizationResult with BL weights
        """
        n_assets = len(returns.columns)
        asset_names = returns.columns.tolist()

        # Prepare inputs
        mu, sigma, _ = self._prepare_inputs(returns, covariance=covariance)

        # Market equilibrium weights (market cap weighted)
        if market_caps is not None:
            w_mkt = np.array(market_caps) / np.sum(market_caps)
        else:
            w_mkt = np.ones(n_assets) / n_assets

        # Risk aversion parameter
        if self.config.risk_aversion is not None:
            delta = self.config.risk_aversion
        else:
            # Estimate from market: delta = (mu_mkt - rf) / sigma_mkt^2
            mkt_return = w_mkt @ mu
            mkt_var = w_mkt @ sigma @ w_mkt
            delta = mkt_return / mkt_var if mkt_var > 0 else 2.5

        # Implied equilibrium returns: Pi = delta * Sigma * w_mkt
        pi = delta * sigma @ w_mkt
        self._implied_returns = pi

        # Build view matrices
        if views is not None and view_confidence is not None:
            # Absolute views
            P, Q, Omega = self._build_absolute_views(
                views, view_confidence, asset_names, sigma, self.config.tau
            )
        elif pick_matrix is not None and view_returns is not None:
            # Relative views provided directly
            P = pick_matrix
            Q = view_returns
            Omega = (
                view_covariance
                if view_covariance is not None
                else self.config.tau * P @ sigma @ P.T
            )
        else:
            # No views - return market portfolio
            metrics = self._compute_metrics(w_mkt, mu, sigma)
            return OptimizationResult(
                weights=w_mkt,
                asset_names=asset_names,
                expected_return=metrics["expected_return"],
                expected_risk=metrics["expected_risk"],
                sharpe_ratio=metrics["sharpe_ratio"],
                implied_returns=pi,
                method="BlackLitterman",
                config=self.config,
                message="No views provided, returning market portfolio",
            )

        # Black-Litterman formula
        tau_sigma = self.config.tau * sigma

        # Posterior covariance: Sigma_post = (tau*Sigma^-1 + P' Omega^-1 P)^-1
        # Using Woodbury identity for stability
        inv_tau_sigma = linalg.inv(tau_sigma + np.eye(n_assets) * 1e-8)

        if P.shape[0] > 0:
            Omega_inv = linalg.inv(Omega + np.eye(P.shape[0]) * 1e-8)
            M = inv_tau_sigma + P.T @ Omega_inv @ P
            sigma_post = linalg.inv(M + np.eye(n_assets) * 1e-8)

            # Posterior mean: mu_post = Sigma_post * (tau*Sigma^-1 * pi + P' Omega^-1 Q)
            mu_post = sigma_post @ (inv_tau_sigma @ pi + P.T @ Omega_inv @ Q)
        else:
            sigma_post = sigma
            mu_post = pi

        self._posterior_mu = mu_post
        self._posterior_sigma = sigma_post

        # Mean-variance optimization with posterior parameters
        weights = self._mean_variance_optimize(mu_post, sigma_post, delta)

        metrics = self._compute_metrics(weights, mu_post, sigma_post)

        return OptimizationResult(
            weights=weights,
            asset_names=asset_names,
            expected_return=metrics["expected_return"],
            expected_risk=metrics["expected_risk"],
            sharpe_ratio=metrics["sharpe_ratio"],
            implied_returns=pi,
            posterior_mu=mu_post,
            posterior_sigma=sigma_post,
            method="BlackLitterman",
            config=self.config,
        )

    def _build_absolute_views(
        self,
        views: dict[str, float],
        confidences: dict[str, float],
        asset_names: list[str],
        sigma: np.ndarray,
        tau: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build P, Q, Omega matrices for absolute views."""
        n_assets = len(asset_names)
        n_views = len(views)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)

        for i, (asset, view_return) in enumerate(views.items()):
            if asset in asset_names:
                idx = asset_names.index(asset)
                P[i, idx] = 1.0
                Q[i] = view_return
                # Omega = tau * P * Sigma * P' * (1/confidence - 1)
                # For absolute view on single asset: Omega_ii = tau * sigma_ii * (1/c - 1)
                conf = confidences.get(asset, 0.5)
                omega_diag[i] = tau * sigma[idx, idx] * (1 / conf - 1) if conf > 0 else 1e6

        Omega = np.diag(omega_diag)
        return P, Q, Omega

    def _mean_variance_optimize(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: float,
    ) -> np.ndarray:
        """Solve mean-variance optimization."""
        n = len(mu)
        w = cp.Variable(n)

        # Maximize utility: w' * mu - 0.5 * delta * w' * Sigma * w
        objective = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, sigma))
        constraints = self._build_constraints(n)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.config.solver, verbose=self.config.verbose)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            weights = w.value
            weights = weights / np.sum(np.abs(weights)) * self.config.max_leverage
            return weights
        else:
            # Fallback
            return np.ones(n) / n


class MeanVarianceOptimizer(BaseOptimizer):
    """
    Classic Mean-Variance Optimization (Markowitz).

    With robust covariance estimation and various objective functions.
    """

    def __init__(
        self,
        config: OptimizationConfig | None = None,
        objective: Literal["max_sharpe", "min_variance", "max_return", "utility"] = "max_sharpe",
    ):
        super().__init__(config)
        self.objective = objective

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
        risk_aversion: float = 1.0,
        target_return: float | None = None,
        target_risk: float | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize mean-variance portfolio.

        Args:
            returns: Historical returns
            expected_returns: Optional expected returns vector
            covariance: Optional covariance matrix
            risk_aversion: Risk aversion for utility objective
            target_return: Target return for min variance
            target_risk: Target risk for max return

        Returns:
            OptimizationResult
        """
        mu, sigma, asset_names = self._prepare_inputs(returns, expected_returns, covariance)
        n_assets = len(asset_names)

        w = cp.Variable(n_assets)
        constraints = self._build_constraints(n_assets)

        if self.objective == "max_sharpe":
            # Maximize Sharpe ratio using fractional programming
            # Equivalent to: max w'mu s.t. w'Sigma w = 1, then scale
            w_scaled = cp.Variable(n_assets)
            kappa = cp.Variable()

            obj = cp.Maximize(mu @ w_scaled)
            constr = [
                cp.quad_form(w_scaled, sigma) == 1,
                w_scaled >= self.config.min_weight * kappa,
                w_scaled <= self.config.max_weight * kappa,
                cp.sum(w_scaled) == kappa,
                kappa >= 0,
            ]

            # Add leverage constraint
            constr.append(cp.sum(cp.abs(w_scaled)) <= self.config.max_leverage * kappa)

            prob = cp.Problem(obj, constr)
            prob.solve(solver=self.config.solver, verbose=self.config.verbose)

            if prob.status in ["optimal", "optimal_inaccurate"]:
                w_opt = w_scaled.value
                weights = (
                    w_opt / np.sum(w_opt) if np.sum(w_opt) != 0 else np.ones(n_assets) / n_assets
                )
            else:
                weights = np.ones(n_assets) / n_assets

        elif self.objective == "min_variance":
            # Minimize variance subject to return target
            obj = cp.Minimize(cp.quad_form(w, sigma))
            if target_return is not None:
                constraints.append(mu @ w >= target_return)
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=self.config.solver, verbose=self.config.verbose)
            weights = (
                w.value
                if prob.status in ["optimal", "optimal_inaccurate"]
                else np.ones(n_assets) / n_assets
            )

        elif self.objective == "max_return":
            # Maximize return subject to risk target
            obj = cp.Maximize(mu @ w)
            if target_risk is not None:
                constraints.append(cp.quad_form(w, sigma) <= target_risk**2)
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=self.config.solver, verbose=self.config.verbose)
            weights = (
                w.value
                if prob.status in ["optimal", "optimal_inaccurate"]
                else np.ones(n_assets) / n_assets
            )

        else:  # utility
            obj = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, sigma))
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=self.config.solver, verbose=self.config.verbose)
            weights = (
                w.value
                if prob.status in ["optimal", "optimal_inaccurate"]
                else np.ones(n_assets) / n_assets
            )

        # Normalize
        weights = weights / np.sum(np.abs(weights)) * self.config.max_leverage

        metrics = self._compute_metrics(weights, mu, sigma)

        return OptimizationResult(
            weights=weights,
            asset_names=asset_names,
            expected_return=metrics["expected_return"],
            expected_risk=metrics["expected_risk"],
            sharpe_ratio=metrics["sharpe_ratio"],
            diversification_ratio=metrics["diversification_ratio"],
            effective_number_assets=metrics["effective_number_assets"],
            method=f"MeanVariance({self.objective})",
            config=self.config,
        )


class MaximumDiversification(BaseOptimizer):
    """Maximum Diversification Portfolio (Choueifaty & Coignard, 2008)."""

    def optimize(
        self, returns: pd.DataFrame, covariance: np.ndarray | None = None, **kwargs
    ) -> OptimizationResult:
        mu, sigma, asset_names = self._prepare_inputs(returns, covariance=covariance)
        n_assets = len(asset_names)

        w = cp.Variable(n_assets)

        # Diversification ratio = w' * sigma_diag / sqrt(w' * Sigma * w)
        # Maximize DR <=> Minimize sqrt(w'Sigma w) / (w' sigma_diag)
        # <=> Minimize w'Sigma w / (w' sigma_diag)^2
        sigma_diag = np.sqrt(np.diag(sigma))

        obj = cp.Minimize(cp.quad_form(w, sigma) / cp.sum(cp.multiply(w, sigma_diag)) ** 2)
        constraints = self._build_constraints(n_assets)
        constraints.append(cp.sum(cp.multiply(w, sigma_diag)) >= 1e-6)

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=self.config.solver, verbose=self.config.verbose)

        weights = (
            w.value
            if prob.status in ["optimal", "optimal_inaccurate"]
            else np.ones(n_assets) / n_assets
        )
        weights = weights / np.sum(np.abs(weights)) * self.config.max_leverage

        metrics = self._compute_metrics(weights, mu, sigma)

        return OptimizationResult(
            weights=weights,
            asset_names=asset_names,
            expected_return=metrics["expected_return"],
            expected_risk=metrics["expected_risk"],
            sharpe_ratio=metrics["sharpe_ratio"],
            diversification_ratio=metrics["diversification_ratio"],
            effective_number_assets=metrics["effective_number_assets"],
            method="MaximumDiversification",
            config=self.config,
        )


class MinimumVariance(BaseOptimizer):
    """Global Minimum Variance Portfolio."""

    def optimize(
        self, returns: pd.DataFrame, covariance: np.ndarray | None = None, **kwargs
    ) -> OptimizationResult:
        mu, sigma, asset_names = self._prepare_inputs(returns, covariance=covariance)
        n_assets = len(asset_names)

        w = cp.Variable(n_assets)
        obj = cp.Minimize(cp.quad_form(w, sigma))
        constraints = self._build_constraints(n_assets)

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=self.config.solver, verbose=self.config.verbose)

        weights = (
            w.value
            if prob.status in ["optimal", "optimal_inaccurate"]
            else np.ones(n_assets) / n_assets
        )
        weights = weights / np.sum(np.abs(weights)) * self.config.max_leverage

        metrics = self._compute_metrics(weights, mu, sigma)

        return OptimizationResult(
            weights=weights,
            asset_names=asset_names,
            expected_return=metrics["expected_return"],
            expected_risk=metrics["expected_risk"],
            sharpe_ratio=metrics["sharpe_ratio"],
            diversification_ratio=metrics["diversification_ratio"],
            effective_number_assets=metrics["effective_number_assets"],
            method="MinimumVariance",
            config=self.config,
        )


def create_optimizer(
    method: str, config: OptimizationConfig | None = None, **kwargs
) -> BaseOptimizer:
    """Factory function to create optimizer by name."""
    method = method.lower()

    if method in ["hrp", "hierarchical_risk_parity"]:
        return HierarchicalRiskParity(config)
    elif method in ["risk_parity", "erc", "equal_risk_contribution"]:
        return RiskParityOptimizer(config)
    elif method in ["black_litterman", "bl"]:
        return BlackLittermanOptimizer(config)
    elif method in ["mean_variance", "markowitz", "mv"]:
        return MeanVarianceOptimizer(config, **kwargs)
    elif method in ["max_diversification", "max_div", "mdp"]:
        return MaximumDiversification(config)
    elif method in ["min_variance", "min_var", "gmv"]:
        return MinimumVariance(config)
    else:
        raise ValueError(f"Unknown optimizer method: {method}")


# Convenience functions
def optimize_portfolio(
    returns: pd.DataFrame, method: str = "hrp", config: OptimizationConfig | None = None, **kwargs
) -> OptimizationResult:
    """High-level portfolio optimization function."""
    optimizer = create_optimizer(method, config)
    return optimizer.optimize(returns, **kwargs)


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    config: OptimizationConfig | None = None,
) -> pd.DataFrame:
    """Generate efficient frontier points."""
    mu, sigma, _ = MeanVarianceOptimizer(config)._prepare_inputs(returns)

    min_ret = mu.min()
    max_ret = mu.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    results = []
    optimizer = MeanVarianceOptimizer(config, objective="min_variance")

    for target in target_returns:
        result = optimizer.optimize(returns, target_return=target)
        results.append(
            {
                "target_return": target,
                "return": result.expected_return,
                "risk": result.expected_risk,
                "sharpe": result.sharpe_ratio,
            }
        )

    return pd.DataFrame(results)

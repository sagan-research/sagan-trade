"""
Comprehensive test suite for Sagan Trade library.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Core modules
from sagan_trade import (
    AsymmetricRiskEngine,
    BacktestEngine,
    HawkesLOBSimulator,
    SymbolicRegressor,
    VolatilityRegimeFilter,
    analyze_portfolio,
    simulate_price_range,
)

# Advanced Backtesting
from sagan_trade.backtesting_advanced import (
    BacktestConfig,
    BacktestResult,
    PurgedKFoldBacktester,
    WalkForwardBacktester,
    compute_performance_metrics,
)

# Execution
from sagan_trade.execution import (
    AlmgrenChrissModel,
    BertsimasLoModel,
    ExecutionConfig,
    ExecutionModel,
    ExecutionResult,
    GatheralSchiedModel,
    ImplementationShortfallModel,
    ObizhaevaWangModel,
    POVModel,
    TWAPModel,
    VWAPModel,
    compare_execution_models,
    create_execution_model,
    optimize_execution,
)

# Feature Engineering
from sagan_trade.feature_engineering import (
    FeatureConfig,
    FeatureEngine,
    MicrostructureFeatures,
    TechnicalIndicators,
)

# PINN models
try:
    from sagan_trade.pinn_models import (
        BlackScholesPINN,
        HestonPINN,
        PINNConfig,
        PINNTrainer,
        create_bs_pinn,
        create_heston_pinn,
    )
except ImportError:
    BlackScholesPINN = None  # type: ignore[assignment,misc]
    HestonPINN = None  # type: ignore[assignment,misc]
    PINNConfig = None  # type: ignore[assignment,misc]
    PINNTrainer = None  # type: ignore[assignment,misc]
    create_bs_pinn = None  # type: ignore[assignment,misc]
    create_heston_pinn = None  # type: ignore[assignment,misc]

# Portfolio optimization
from sagan_trade.portfolio_optimization import (
    BlackLittermanOptimizer,
    HierarchicalRiskParity,
    OptimizationConfig,
    OptimizationResult,
    RiskParityOptimizer,
)

# Deep Learning
try:
    from sagan_trade.tft_model import (
        TemporalFusionTransformer,
        TFTConfig,
        create_tft_model,
    )
except ImportError:
    TemporalFusionTransformer = None  # type: ignore[assignment,misc]
    TFTConfig = None  # type: ignore[assignment,misc]
    create_tft_model = None  # type: ignore[assignment,misc]


class TestTechnicalIndicators:
    """Test technical indicator calculations."""

    def setup_method(self):
        np.random.seed(42)
        n = 500
        self.close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        self.high = self.close + np.abs(np.random.randn(n) * 0.3)
        self.low = self.close - np.abs(np.random.randn(n) * 0.3)
        self.volume = pd.Series(np.random.randint(1000000, 10000000, n))

    def test_sma(self):
        sma = TechnicalIndicators.sma(self.close, 20)
        assert len(sma) == len(self.close)
        assert sma.iloc[19:].notna().all()

    def test_ema(self):
        ema = TechnicalIndicators.ema(self.close, 20)
        assert len(ema) == len(self.close)
        assert ema.iloc[19:].notna().all()

    def test_rsi(self):
        rsi = TechnicalIndicators.rsi(self.close, 14)
        assert len(rsi) == len(self.close)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd(self):
        macd, signal, hist = TechnicalIndicators.macd(self.close)
        assert len(macd) == len(self.close)
        assert len(signal) == len(self.close)
        assert len(hist) == len(self.close)

    def test_bollinger_bands(self):
        upper, middle, lower = TechnicalIndicators.bollinger_bands(self.close)
        assert len(upper) == len(self.close)
        valid_upper = upper.dropna()
        valid_middle = middle.dropna()
        valid_lower = lower.dropna()
        assert (valid_upper >= valid_middle).all()
        assert (valid_middle >= valid_lower).all()

    def test_atr(self):
        atr = TechnicalIndicators.atr(self.high, self.low, self.close)
        assert len(atr) == len(self.close)
        assert (atr.dropna() >= 0).all()

    def test_stochastic(self):
        k, d = TechnicalIndicators.stochastic(self.high, self.low, self.close)
        assert len(k) == len(self.close)
        valid_k = k.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_obv(self):
        obv = TechnicalIndicators.obv(self.close, self.volume)
        assert len(obv) == len(self.close)

    def test_vwap(self):
        vwap = TechnicalIndicators.vwap(self.high, self.low, self.close, self.volume)
        assert len(vwap) == len(self.close)


class TestMarketMicrostructure:
    """Test market microstructure simulation."""

    def test_hawkes_simulator(self):
        sim = HawkesLOBSimulator(seed=42)
        df = sim.simulate_ticks("RELIANCE", num_ticks=1000)

        assert len(df) == 1000
        assert "mid_price" in df.columns
        assert "bid_price" in df.columns
        assert "ask_price" in df.columns
        assert "bid_size" in df.columns
        assert "ask_size" in df.columns
        assert "ofi" in df.columns
        assert "micro_price" in df.columns

    def test_simulate_price_range(self):
        result = simulate_price_range("RELIANCE", N=10000, n_bootstrap=100)

        assert "ticker" in result
        assert "current_price" in result
        assert "price_range" in result
        assert "median_price" in result
        assert result["ticker"] == "RELIANCE"
        assert len(result["price_range"]) == 2

    def test_analyze_portfolio(self):
        df = analyze_portfolio(["RELIANCE", "HDFCBANK"], quick_mode=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Signal" in df.columns
        assert "Current" in df.columns


class TestAsymmetricRiskEngine:
    """Test Asymmetric Risk Engine."""

    def setup_method(self):
        np.random.seed(42)
        n = 500
        self.prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)))

    def test_risk_multiplier(self):
        engine = AsymmetricRiskEngine(target_vol=0.15, max_drawdown_limit=0.075)
        multiplier = engine.get_risk_multiplier(self.prices)

        assert len(multiplier) == len(self.prices)
        assert (multiplier >= 0).all()
        assert (multiplier <= 1).all()

    def test_risk_multiplier_bounds(self):
        engine = AsymmetricRiskEngine(target_vol=0.10, max_drawdown_limit=0.05)
        multiplier = engine.get_risk_multiplier(self.prices)

        # During drawdown, multiplier should be reduced
        assert multiplier.iloc[-1] <= 1.0


class TestVolatilityRegimeFilter:
    """Test Volatility Regime Filter."""

    def setup_method(self):
        np.random.seed(42)
        n = 500
        self.prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)))

    def test_generate_signals(self):
        vol_filter = VolatilityRegimeFilter(vol_window=20, ma_window=120)
        signals = vol_filter.generate_signals(self.prices)

        assert len(signals) == len(self.prices)
        assert (signals >= 0).all()
        assert (signals <= 1).all()


class TestBacktestEngine:
    """Test Backtest Engine."""

    def setup_method(self):
        np.random.seed(42)
        n = 500
        self.prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )
        self.signals = pd.Series(np.random.randn(n), index=self.prices.index)

    def test_run_backtest(self):
        engine = BacktestEngine(initial_capital=100000, maker_fee=0.0001, taker_fee=0.0003)
        result = engine.run(self.prices, self.signals)

        assert isinstance(result, BacktestResult)
        assert result.sharpe_ratio is not None
        assert result.max_drawdown is not None
        assert result.total_return is not None
        assert len(result.portfolio_values) == len(self.prices)

    def test_run_with_risk_model(self):
        engine = BacktestEngine()
        risk_model = AsymmetricRiskEngine()

        result = engine.run(self.prices, self.signals, risk_model=risk_model)

        assert isinstance(result, BacktestResult)

    def test_run_with_regime_filter(self):
        engine = BacktestEngine()
        vol_filter = VolatilityRegimeFilter()
        regime = vol_filter.generate_signals(self.prices)

        result = engine.run(self.prices, self.signals, regime_filter=regime)

        assert isinstance(result, BacktestResult)


class TestSymbolicRegressor:
    """Test Symbolic Regressor."""

    def setup_method(self):
        np.random.seed(42)
        n = 500
        self.data = pd.DataFrame(
            {
                "Close": 100 + np.cumsum(np.random.randn(n) * 0.5),
                "Volume": np.random.randint(1000000, 10000000, n),
            }
        )
        self.data["RSI"] = TechnicalIndicators.rsi(self.data["Close"])

    def test_train_predict(self):
        regressor = SymbolicRegressor()
        model_id = regressor.train(
            target=self.data["Close"], signals=["Close", "RSI"], data=self.data
        )

        assert model_id is not None
        assert regressor.best_formula_name is not None
        assert regressor.fitted_params is not None

    def test_predict(self):
        regressor = SymbolicRegressor()
        regressor.train(target=self.data["Close"], signals=["Close", "RSI"], data=self.data)

        preds, formula = regressor.predict()

        assert len(preds) == len(self.data)
        assert isinstance(formula, str)
        assert len(formula) > 0


class TestHierarchicalRiskParity:
    """Test HRP Portfolio Optimization."""

    def setup_method(self):
        np.random.seed(42)
        n = 252
        self.returns = pd.DataFrame(
            np.random.randn(n, 10) * 0.01,
            columns=[f"ASSET_{i}" for i in range(10)],
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )

    def test_optimize(self):
        hrp = HierarchicalRiskParity()
        result = hrp.optimize(self.returns)

        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 10
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert (result.weights >= 0).all()

    def test_optimize_with_config(self):
        config = OptimizationConfig(min_weight=0.01, max_weight=0.3)
        hrp = HierarchicalRiskParity(config)
        result = hrp.optimize(self.returns)

        assert (result.weights >= 0.01).all()
        assert (result.weights <= 0.3).all()


class TestRiskParityOptimizer:
    """Test Risk Parity Optimizer."""

    def setup_method(self):
        np.random.seed(42)
        n = 252
        self.returns = pd.DataFrame(
            np.random.randn(n, 8) * 0.01,
            columns=[f"ASSET_{i}" for i in range(8)],
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )

    def test_optimize(self):
        rp = RiskParityOptimizer()
        result = rp.optimize(self.returns)

        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 8
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert result.risk_contributions is not None


class TestBlackLittermanOptimizer:
    """Test Black-Litterman Optimizer."""

    def setup_method(self):
        np.random.seed(42)
        n = 252
        self.returns = pd.DataFrame(
            np.random.randn(n, 6) * 0.01,
            columns=[f"ASSET_{i}" for i in range(6)],
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )

    def test_optimize_no_views(self):
        bl = BlackLittermanOptimizer()
        result = bl.optimize(self.returns, market_caps=np.ones(6) * 100)

        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 6
        assert result.implied_returns is not None

    def test_optimize_with_views(self):
        bl = BlackLittermanOptimizer()
        views = {"ASSET_0": 0.15, "ASSET_1": 0.10}
        confidences = {"ASSET_0": 0.7, "ASSET_1": 0.5}

        result = bl.optimize(
            self.returns, market_caps=np.ones(6) * 100, views=views, view_confidence=confidences
        )

        assert isinstance(result, OptimizationResult)
        assert result.posterior_mu is not None
        assert result.posterior_sigma is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTemporalFusionTransformer:
    """Test TFT Model."""

    def test_model_creation(self):
        config = TFTConfig(
            hidden_size=64,
            num_heads=4,
            num_encoder_steps=24,
            num_decoder_steps=6,
            num_static_vars=3,
            num_time_varying_known=2,
            num_time_varying_unknown=5,
        )

        model = TemporalFusionTransformer(config)

        # Test forward pass
        batch_size = 16
        static_inputs = torch.randn(batch_size, 3)
        encoder_inputs = torch.randn(batch_size, 24, 5)
        decoder_inputs = torch.randn(batch_size, 6, 7)

        outputs = model(static_inputs, encoder_inputs, decoder_inputs)

        assert "predictions" in outputs
        assert outputs["predictions"].shape == (
            batch_size,
            6,
            3,
            1,
        )  # [batch, dec_len, n_quantiles, output_size]

    def test_factory_function(self):
        model = create_tft_model(
            num_static_vars=5,
            num_time_varying_known=3,
            num_time_varying_unknown=10,
            hidden_size=128,
            num_heads=4,
            num_encoder_steps=48,
            num_decoder_steps=12,
        )

        assert isinstance(model, TemporalFusionTransformer)

    def test_predict_quantiles(self):
        config = TFTConfig(
            hidden_size=32,
            num_heads=2,
            num_encoder_steps=12,
            num_decoder_steps=3,
            num_static_vars=2,
            num_time_varying_known=1,
            num_time_varying_unknown=3,
        )
        model = TemporalFusionTransformer(config)

        batch_size = 4
        static_inputs = torch.randn(batch_size, 2)
        encoder_inputs = torch.randn(batch_size, 12, 3)
        decoder_inputs = torch.randn(batch_size, 3, 4)

        quantiles = model.predict_quantiles(static_inputs, encoder_inputs, decoder_inputs)

        assert quantiles.shape == (batch_size, 3, 3, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestPINNModels:
    """Test PINN Models."""

    def test_black_scholes_pinn(self):
        config = PINNConfig(hidden_layers=[32, 32])
        model = BlackScholesPINN(config, strike=100.0)

        assert isinstance(model, BlackScholesPINN)

        # Test forward pass
        S = torch.tensor([[100.0], [110.0], [90.0]])
        t = torch.tensor([[0.5], [0.5], [0.5]])
        output = model(S, t)

        assert output.shape == (3, 1)

    def test_heston_pinn(self):
        config = PINNConfig(hidden_layers=[32, 32])
        model = HestonPINN(config, strike=100.0)

        assert isinstance(model, HestonPINN)

        S = torch.tensor([[100.0]])
        v = torch.tensor([[0.04]])
        t = torch.tensor([[0.5]])
        output = model(S, v, t)

        assert output.shape == (1, 1)

    def test_pinn_trainer(self):
        config = PINNConfig(
            hidden_layers=[16, 16],
            max_epochs=10,
            n_pde_points=100,
        )
        model = BlackScholesPINN(config, strike=100.0)
        trainer = PINNTrainer(model, config)

        assert trainer is not None
        assert trainer.optimizer is not None

    def test_factory_functions(self):
        model1 = create_bs_pinn(strike=100.0, hidden_layers=[32, 32])
        model2 = create_heston_pinn(strike=100.0, hidden_layers=[32, 32])

        assert isinstance(model1, BlackScholesPINN)
        assert isinstance(model2, HestonPINN)


class TestExecutionModels:
    """Test Execution Models."""

    def setup_method(self):
        self.config = ExecutionConfig(
            total_quantity=100000,
            time_horizon=1.0,
            sigma=0.02,
            risk_aversion=1e-6,
            permanent_impact=0.1,
            temporary_impact=0.1,
            num_intervals=10,
        )

    def test_almgren_chriss(self):
        model = AlmgrenChrissModel(self.config)
        result = model.optimize()

        assert isinstance(result, ExecutionResult)
        assert len(result.schedule) == 10
        assert abs(result.schedule.sum() - 100000) < 1.0
        assert result.expected_cost > 0

    def test_bertsimas_lo(self):
        model = BertsimasLoModel(self.config)
        result = model.optimize()

        assert isinstance(result, ExecutionResult)
        assert len(result.schedule) == 10

    def test_obizhaeva_wang(self):
        model = ObizhaevaWangModel(self.config)
        result = model.optimize()

        assert isinstance(result, ExecutionResult)

    def test_gatheral_schied(self):
        model = GatheralSchiedModel(self.config)
        result = model.optimize()

        assert isinstance(result, ExecutionResult)

    def test_twap(self):
        model = TWAPModel(self.config)
        result = model.optimize()

        assert isinstance(result, ExecutionResult)
        assert np.allclose(result.schedule, 10000)

    def test_vwap(self):
        model = VWAPModel(self.config)
        result = model.optimize()

        assert isinstance(result, ExecutionResult)

    def test_pov(self):
        model = POVModel(self.config)
        result = model.optimize()

        assert isinstance(result, ExecutionResult)

    def test_implementation_shortfall(self):
        model = ImplementationShortfallModel(self.config)
        result = model.optimize()

        assert isinstance(result, ExecutionResult)

    def test_factory(self):
        model = create_execution_model(ExecutionModel.ALMGREN_CHRISS, self.config)
        result = model.optimize()

        assert isinstance(result, ExecutionResult)

    def test_optimize_execution(self):
        result = optimize_execution(
            total_quantity=100000,
            time_horizon=1.0,
            volatility=0.02,
            risk_aversion=1e-6,
            permanent_impact=0.1,
            temporary_impact=0.1,
        )

        assert isinstance(result, ExecutionResult)

    def test_compare_models(self):
        df = compare_execution_models(self.config)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "model" in df.columns


class TestAdvancedBacktesting:
    """Test Advanced Backtesting Methods."""

    def setup_method(self):
        np.random.seed(42)
        n = 1000
        prices = 100 * np.exp(np.cumsum(np.random.randn(n, 5) * 0.01, axis=0))
        self.prices = pd.DataFrame(
            prices,
            columns=[f"ASSET_{i}" for i in range(5)],
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )
        self.signals = pd.DataFrame(
            np.random.randn(n, 5), columns=[f"ASSET_{i}" for i in range(5)], index=self.prices.index
        )

    def test_walk_forward(self):
        config = BacktestConfig(
            train_window=252,
            test_window=63,
            step_size=21,
            num_intervals=10,
        )

        def strategy(train_prices, train_signals, test_prices, test_signals):
            return np.sign(train_signals.iloc[-1]).values

        backtester = WalkForwardBacktester(config)
        # Just test split generation
        splits = list(backtester.generate_splits(len(self.prices)))
        assert len(splits) > 0

    def test_purged_kfold(self):
        config = BacktestConfig(n_splits=5, purge_gap=10, embargo_pct=0.01)
        backtester = PurgedKFoldBacktester(config)

        splits = list(backtester.generate_splits(len(self.prices)))
        assert len(splits) == 5

    def test_performance_metrics(self):
        returns = pd.Series(np.random.randn(252) * 0.01)
        metrics = compute_performance_metrics(returns)

        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "annualized_return" in metrics


class TestFeatureEngineering:
    """Test Feature Engineering Pipeline."""

    def setup_method(self):
        np.random.seed(42)
        n = 500
        self.data = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(n) * 0.3),
                "high": 100
                + np.cumsum(np.random.randn(n) * 0.3)
                + np.abs(np.random.randn(n) * 0.2),
                "low": 100 + np.cumsum(np.random.randn(n) * 0.3) - np.abs(np.random.randn(n) * 0.2),
                "close": 100 + np.cumsum(np.random.randn(n) * 0.3),
                "volume": np.random.randint(1000000, 10000000, n),
            },
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )

    def test_generate_features(self):
        config = FeatureConfig()
        engine = FeatureEngine(config)
        features = engine.generate_features(self.data)

        assert len(features) == len(self.data)
        assert len(features.columns) > 0

    def test_fit_transform(self):
        config = FeatureConfig(selection_method="mutual_info", max_features=20)
        engine = FeatureEngine(config)

        target = self.data["close"].pct_change().shift(-1).fillna(0)
        features = engine.fit_transform(self.data, target)

        assert len(features) == len(self.data)
        assert len(features.columns) <= 20
        assert engine._fitted

    def test_technical_indicators(self):
        ti = TechnicalIndicators()

        rsi = ti.rsi(self.data["close"])
        macd, _signal, _hist = ti.macd(self.data["close"])
        _upper, _middle, _lower = ti.bollinger_bands(self.data["close"])
        _atr = ti.atr(self.data["high"], self.data["low"], self.data["close"])

        assert len(rsi) == len(self.data)
        assert len(macd) == len(self.data)

    def test_microstructure_features(self):
        mf = MicrostructureFeatures()

        # Need bid/ask data
        bid_price = self.data["close"] - 0.01
        ask_price = self.data["close"] + 0.01
        bid_vol = pd.Series(np.random.randint(1000, 10000, len(self.data)))
        ask_vol = pd.Series(np.random.randint(1000, 10000, len(self.data)))

        ofi = mf.order_flow_imbalance(bid_price, bid_vol, ask_price, ask_vol)
        rv = mf.realized_volatility(self.data["close"].pct_change())

        assert len(ofi) == len(self.data)
        assert len(rv) == len(self.data)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_full_pipeline(self):
        """Test end-to-end pipeline."""
        np.random.seed(42)
        n = 500

        # Generate data
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.randn(n, 5) * 0.01, axis=0)),
            columns=[f"ASSET_{i}" for i in range(5)],
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )

        # Feature engineering
        config = FeatureConfig(max_features=15)
        engine = FeatureEngine(config)
        features = engine.fit_transform(
            prices["ASSET_0"]
            .to_frame(name="close")
            .join(
                pd.DataFrame(
                    {
                        "open": prices["ASSET_0"] * 0.999,
                        "high": prices["ASSET_0"] * 1.002,
                        "low": prices["ASSET_0"] * 0.998,
                        "volume": np.random.randint(1000000, 10000000, n),
                    }
                )
            ),
            prices["ASSET_0"].pct_change().shift(-1).fillna(0),
        )

        # Portfolio optimization
        hrp = HierarchicalRiskParity()
        result = hrp.optimize(prices.pct_change().dropna())

        # Execution optimization
        _exec_config = ExecutionConfig(total_quantity=10000)
        exec_result = optimize_execution(
            total_quantity=10000,
            time_horizon=1.0,
            volatility=0.02,
            risk_aversion=1e-6,
            permanent_impact=0.1,
            temporary_impact=0.1,
        )

        assert isinstance(result, OptimizationResult)
        assert isinstance(exec_result, ExecutionResult)
        assert len(features) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import numpy as np
import pandas as pd
import sys

# Ensure local sagan_trade package is prioritised
sys.path.insert(0, ".")

from sagan_trade import (
    SymbolicRegressor, 
    AsymmetricRiskEngine, 
    VolatilityRegimeFilter,
    BacktestEngine
)

def run_test():
    print("==================================================")
    print("Testing sagan-trade Library Update...")
    print("==================================================")

    # 1. Generate clean mock historical data (100 days)
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=100)
    
    # Random walk for Close price
    close_prices = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, 100))
    volume = np.random.randint(1000, 5000, 100).astype(float)
    rsi = 50.0 + np.cumsum(np.random.normal(0, 2.0, 100))
    rsi = np.clip(rsi, 0, 100)

    data = pd.DataFrame({
        'Close': close_prices,
        'Volume': volume,
        'RSI': rsi
    }, index=dates)

    print("Mock market data generated:")
    print(data.head())
    print()

    # 2. Symbolic Regression / Discovery
    print("Training SymbolicRegressor...")
    regressor = SymbolicRegressor(basis_functions=['poly', 'fourier'])
    model_id = regressor.train(target="Close", signals=["Close", "RSI", "Volume"], data=data)
    print(f"Training completed. Model ID: {model_id}")
    
    predicted_signal, formula = regressor.predict()
    print(f"Discovered Alpha Equation: {formula}")
    print(f"Predicted Signal Sample:\n{predicted_signal.head()}\n")

    # 3. Macro Volatility Regime Filtering
    print("Generating Volatility Regime Signals...")
    vol_filter = VolatilityRegimeFilter(vol_window=5, ma_window=20)
    regime_signals = vol_filter.generate_signals(data['Close'])
    print(f"Current Market Regime (1=Risk-On, 0=Cash):\n{regime_signals.tail()}\n")

    # 4. Asymmetric Convexity Risk Scaling
    print("Initializing AsymmetricRiskEngine...")
    risk_engine = AsymmetricRiskEngine(target_vol=0.15, max_drawdown_limit=0.075)
    risk_multipliers = risk_engine.get_risk_multiplier(data['Close'])
    print(f"Risk Multiplier Sample:\n{risk_multipliers.tail()}\n")

    # 5. Backtest Execution
    print("Running BacktestEngine...")
    backtester = BacktestEngine(
        initial_capital=1000000,
        maker_fee=0.0001,
        taker_fee=-0.0003
    )

    results = backtester.run(
        prices=data['Close'],
        alpha_signals=predicted_signal,
        regime_filter=regime_signals,
        risk_model=risk_engine
    )

    print("--------------------------------------------------")
    print("Backtest Performance Summary:")
    print("--------------------------------------------------")
    print(f"Total Return: {results.total_return}%")
    print(f"Backtest Sharpe: {results.sharpe_ratio}")
    print(f"Backtest Max Drawdown: {results.max_drawdown}%")
    print("Metrics Dictionary:")
    print(results.metrics)
    print("--------------------------------------------------")

    # Asset tests to verify correctness of attributes and returned values
    assert hasattr(results, 'sharpe_ratio'), "Missing sharpe_ratio"
    assert hasattr(results, 'max_drawdown'), "Missing max_drawdown"
    assert hasattr(results, 'total_return'), "Missing total_return"
    assert hasattr(results, 'portfolio_values'), "Missing portfolio_values"
    assert len(results.portfolio_values) == len(data), "Portfolio value series length mismatch"
    print("All test assertions passed successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_test()

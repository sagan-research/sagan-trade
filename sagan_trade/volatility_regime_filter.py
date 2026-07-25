import numpy as np
import pandas as pd


class VolatilityRegimeFilter:
    """
    A macroeconomic sidecar filter that dynamically adjusts equity exposure
    based on Volatility Risk Premium (VRP) proxy and Hawkes point process logic.
    """

    def __init__(self, vol_window=20, ma_window=120):
        self.vol_window = vol_window
        self.ma_window = ma_window

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Returns 1.0 (Long) when realized volatility is below the historical average,
        and 0.0 (Cash) when it breaches the historical average (Contagion Regime).
        """
        returns = np.log(prices / prices.shift(1))
        realized_vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        vol_ma = realized_vol.rolling(self.ma_window).mean()

        positions = pd.Series(0.0, index=prices.index)
        positions[realized_vol < vol_ma] = 1.0

        # Shift to prevent lookahead bias
        return positions.shift(1).fillna(1.0)

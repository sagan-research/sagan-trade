import numpy as np
import pandas as pd


class AsymmetricRiskEngine:
    """
    Asymmetric Convexity Risk Engine that dynamically scales exposure.

    This engine calculates a position scaling multiplier based on target
    volatility and maximum drawdown budgets to proactively manage risk.

    Args:
        target_vol (float, optional): The annualized target volatility budget. Defaults to 0.15 (15%).
        max_drawdown_limit (float, optional): The maximum allowable drawdown limit. Defaults to 0.075 (7.5%).
    """

    def __init__(self, target_vol: float = 0.15, max_drawdown_limit: float = 0.075) -> None:
        self.target_vol: float = target_vol
        self.max_drawdown_limit: float = max_drawdown_limit

    def get_risk_multiplier(self, prices: pd.Series) -> pd.Series:
        """
        Calculate a dynamic position scaling multiplier.

        This method computes a multiplier bounded between [0.0, 1.0] by evaluating
        the rolling 20-day realized volatility and the running cumulative drawdown
        against the engine's target budgets. The blended multiplier reduces exposure
        as risk limits are approached.

        Args:
            prices (pd.Series): A pandas Series of asset prices indexed by time.

        Returns:
            pd.Series: A pandas Series of risk multipliers bounded in [0.0, 1.0].

        Examples:
            >>> import pandas as pd
            >>> engine = AsymmetricRiskEngine(target_vol=0.15, max_drawdown_limit=0.10)
            >>> prices = pd.Series([100, 95, 90, 85, 80])
            >>> multipliers = engine.get_risk_multiplier(prices)
            >>> multipliers.iloc[-1] < 1.0
            True
        """
        # Calculate daily log returns
        returns = np.log(prices / prices.shift(1)).fillna(0.0)

        # Calculate 20-day rolling annualized volatility
        rolling_std = returns.rolling(window=20).std()
        rolling_vol = rolling_std * np.sqrt(252)
        rolling_vol = rolling_vol.fillna(self.target_vol)

        # Calculate running cumulative drawdown of the underlying asset
        cum_prices = prices / prices.iloc[0]
        running_max = cum_prices.cummax()
        drawdown = (cum_prices - running_max) / running_max
        abs_drawdown = np.abs(drawdown)

        # Volatility Scaling Multiplier: reduce exposure if realized vol exceeds target vol
        # Multiplier = Target Vol / Realized Vol
        vol_multiplier = self.target_vol / (rolling_vol + 1e-8)
        vol_multiplier = np.clip(vol_multiplier, 0.0, 1.0)

        # Drawdown Scaling Multiplier: reduce exposure linearly as drawdown approaches max drawdown limit
        # Multiplier = (Max DD - Realized DD) / Max DD
        dd_multiplier = (self.max_drawdown_limit - abs_drawdown) / (self.max_drawdown_limit + 1e-8)
        dd_multiplier = np.clip(dd_multiplier, 0.0, 1.0)

        # Blended risk scaling multiplier
        multiplier = vol_multiplier * dd_multiplier
        return pd.Series(multiplier, index=prices.index).fillna(1.0)

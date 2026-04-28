import numpy as np
import pandas as pd

class RiskEngine:
    """
    Advanced Risk Management for Symbolic Portfolios.
    Focuses on Volatility Targeting and Drawdown Control.
    """
    def __init__(self, target_vol=0.10, max_drawdown_limit=0.075):
        self.target_vol = target_vol
        self.max_drawdown_limit = max_drawdown_limit

    def calculate_vol_scale(self, returns, lookback=20):
        """
        Calculates a scaling factor to target a specific annualized volatility.
        """
        if len(returns) < lookback:
            return 1.0
        
        realized_vol = returns.rolling(window=lookback).std() * np.sqrt(252)
        scale = self.target_vol / (realized_vol + 1e-9)
        return scale.fillna(1.0)

    def apply_risk_parity(self, returns_dict):
        """
        Calculates Inverse-Variance weights for a portfolio of returns.
        """
        vols = {}
        for ticker, rets in returns_dict.items():
            vols[ticker] = np.std(rets) + 1e-9
            
        inv_vols = {t: 1.0/v for t, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        weights = {t: iv/total_inv_vol for t, iv in inv_vols.items()}
        return weights

    def dynamic_position_sizing(self, signal, price_vol):
        """
        Adjusts signal strength based on price volatility.
        """
        # Simple Vol Damping: Signal / Vol
        return signal / (price_vol + 1e-9)

    def dynamic_vol_target(self, market_vol, base_target=0.12):
        """
        Adjusts target volatility based on market-wide volatility (Regime Awareness).
        """
        regime_factor = np.exp(-max(0, market_vol - 0.20) * 2.0)
        return base_target * regime_factor

    def drawdown_protector(self, current_drawdown, base_target=0.12):
        """
        Aggressively reduces target volatility as the portfolio approaches the limit.
        """
        buffer = max(0, self.max_drawdown_limit - abs(current_drawdown))
        scale = buffer / self.max_drawdown_limit
        return base_target * (scale**2)

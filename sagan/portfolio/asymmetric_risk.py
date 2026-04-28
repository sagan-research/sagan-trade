import numpy as np
import pandas as pd
from sagan.portfolio.risk_engine import RiskEngine

class AsymmetricRiskEngine(RiskEngine):
    """
    Asymmetric Risk Management inspired by high-fidelity market models.
    Rides upside volatility while aggressively downsizing on downside risk.
    """
    def __init__(self, target_vol=0.15, max_drawdown_limit=0.075):
        super().__init__(target_vol, max_drawdown_limit)

    def calculate_asymmetric_scale(self, returns, current_price, ma_20, lookback=20):
        """
        Scales position based on the asymmetry of returns.
        Increases exposure during upside volatility and cuts it during downside.
        """
        if len(returns) < lookback:
            return 1.0
            
        # 1. Downside Volatility (Semi-deviation)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) + 1e-9
        
        # 2. Upside Volatility
        upside_returns = returns[returns > 0]
        upside_vol = np.std(upside_returns) * np.sqrt(252) + 1e-9
        
        # 3. Momentum Multiplier
        # If price > MA20, we are in an upside regime
        momentum_factor = 1.25 if current_price > ma_20 else 0.5
        
        # 4. Asymmetric Scaling
        # We target vol based on downside risk. If downside risk is low, we can lever up.
        # But we only do so if momentum is positive.
        scale = (self.target_vol / downside_vol) * momentum_factor
        
        # Cap leverage at 3.0x for aggressive capture
        return min(scale, 3.0)

    def downside_convexity(self, returns, current_price, ma_20):
        """
        Frontier Research: Implements Downside Convexity Adjustment.
        Creates a convex exposure profile that levers up on positive skewness 
        and de-leverages on downside semi-variance.
        """
        downside_rets = returns[returns < 0]
        if len(downside_rets) < 2:
            semi_var = 1e-6 # Default small variance if no down days
        else:
            semi_var = np.var(downside_rets) + 1e-9
        
        # Convex Momentum Factor
        # Using exponential scaling for positive momentum
        momentum = (current_price / ma_20) - 1
        convex_factor = np.exp(momentum * 5.0) # Sharp increase on upside
        
        # Scale by Inverse Semi-Variance
        return (1.0 / np.sqrt(semi_var)) * convex_factor * 0.01 # Normalizing factor

    def adaptive_kelly(self, win_rate, profit_factor, current_drawdown):
        """
        Calculates a drawdown-constrained Kelly Fraction.
        """
        # Standard Kelly: f = p - (1-p)/b
        # p = win_rate, b = profit_factor
        kelly_f = win_rate - (1 - win_rate) / (profit_factor + 1e-9)
        
        # Constrain Kelly by Drawdown Buffer
        buffer = max(0, self.max_drawdown_limit - abs(current_drawdown))
        protection = (buffer / self.max_drawdown_limit)**2
        
        return max(0, kelly_f * protection * 0.25) # Fractional Kelly (conservative)

    def asymmetric_weighting(self, returns_dict, current_drawdown):
        """
        Portfolio weighting that favors assets with high Upside/Downside ratios.
        """
        scores = {}
        for ticker, rets in returns_dict.items():
            # Cold start check: if history is too short, use equal weighting proxy
            if len(rets) < 5:
                scores[ticker] = 1.0
            else:
                downside = np.std(rets[rets < 0]) + 1e-9
                upside = np.std(rets[rets > 0]) + 1e-9
                scores[ticker] = (upside / downside)
            
        total_score = sum(scores.values())
        weights = {t: s/total_score for t, s in scores.items()}
        
        # Apply Drawdown Protector to the entire portfolio weight
        dd_scale = (max(0, self.max_drawdown_limit - abs(current_drawdown)) / self.max_drawdown_limit)**2
        return {t: w * dd_scale for t, w in weights.items()}


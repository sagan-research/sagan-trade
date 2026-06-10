import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class HawkesLOBSimulator:
    """
    Simulates high-fidelity limit order book (LOB) data for Indian equities (NSE)
    with self-exciting Hawkes process dynamics, queue depth, and spread variations.
    """
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Stock Profiles mapping to NSE market realities
        # Tick size is fixed at ₹0.05 for NSE
        self.stock_profiles = {
            "RELIANCE": {
                "name": "Reliance Industries Ltd (High Liquidity, Large Cap)",
                "base_price": 2500.0,
                "mean_spread_ticks": 4,      # ~0.20
                "min_spread_ticks": 2,       # 0.10
                "volatility_annual": 0.15,
                "base_depth": 10000,
                "hawkes_mu": 0.5,            # Baseline trade intensity
                "hawkes_alpha": 0.3,         # Self-excitation parameter
                "hawkes_beta": 0.8,          # Decay rate
                "tick_size": 0.05
            },
            "HDFCBANK": {
                "name": "HDFC Bank Ltd (High Liquidity, Large Cap)",
                "base_price": 1600.0,
                "mean_spread_ticks": 3,      # ~0.15
                "min_spread_ticks": 1,       # 0.05
                "volatility_annual": 0.12,
                "base_depth": 15000,
                "hawkes_mu": 0.6,
                "hawkes_alpha": 0.35,
                "hawkes_beta": 0.9,
                "tick_size": 0.05
            },
            "INFY": {
                "name": "Infosys Ltd (High Liquidity, Tech Large Cap)",
                "base_price": 1400.0,
                "mean_spread_ticks": 4,      # ~0.20
                "min_spread_ticks": 2,       # 0.10
                "volatility_annual": 0.18,
                "base_depth": 8000,
                "hawkes_mu": 0.4,
                "hawkes_alpha": 0.25,
                "hawkes_beta": 0.7,
                "tick_size": 0.05
            },
            "MRF": {
                "name": "MRF Ltd (Extreme Price, Ultra-Low Depth/Liquidity)",
                "base_price": 120000.0,
                "mean_spread_ticks": 500,    # ~₹25.00
                "min_spread_ticks": 100,     # ~₹5.00
                "volatility_annual": 0.20,
                "base_depth": 5,             # Tiny depth!
                "hawkes_mu": 0.02,           # Rare trades
                "hawkes_alpha": 0.1,
                "hawkes_beta": 0.5,
                "tick_size": 0.05
            },
            "SUZLON": {
                "name": "Suzlon Energy Ltd (Low Price, Massive Queue Liquidity)",
                "base_price": 40.0,
                "mean_spread_ticks": 1,      # Spreads are almost always 1 tick due to constraint
                "min_spread_ticks": 1,
                "volatility_annual": 0.45,
                "base_depth": 500000,        # Massive queue size!
                "hawkes_mu": 1.2,            # Fast execution, but hard to cross
                "hawkes_alpha": 0.4,
                "hawkes_beta": 1.1,
                "tick_size": 0.05
            },
            "TCS": {
                "name": "Tata Consultancy Services Ltd (High Liquidity, Tech Large Cap)",
                "base_price": 3200.0,
                "mean_spread_ticks": 4,      # ~0.20
                "min_spread_ticks": 2,       # 0.10
                "volatility_annual": 0.14,
                "base_depth": 7000,
                "hawkes_mu": 0.45,
                "hawkes_alpha": 0.28,
                "hawkes_beta": 0.75,
                "tick_size": 0.05
            },
            "ICICIBANK": {
                "name": "ICICI Bank Ltd (High Liquidity, Banking Large Cap)",
                "base_price": 900.0,
                "mean_spread_ticks": 3,      # ~0.15
                "min_spread_ticks": 1,       # 0.05
                "volatility_annual": 0.16,
                "base_depth": 12000,
                "hawkes_mu": 0.55,
                "hawkes_alpha": 0.32,
                "hawkes_beta": 0.85,
                "tick_size": 0.05
            },
            "ITC": {
                "name": "ITC Ltd (Ultra-High Liquidity, Low Volatility Large Cap)",
                "base_price": 450.0,
                "mean_spread_ticks": 1,      # ~0.05
                "min_spread_ticks": 1,       # 0.05
                "volatility_annual": 0.08,
                "base_depth": 35000,
                "hawkes_mu": 0.75,
                "hawkes_alpha": 0.2,
                "hawkes_beta": 0.95,
                "tick_size": 0.05
            },
            "ZOMATO": {
                "name": "Zomato Ltd (Mid Cap Growth, High Retail Intensity)",
                "base_price": 180.0,
                "mean_spread_ticks": 2,      # ~0.10
                "min_spread_ticks": 1,       # 0.05
                "volatility_annual": 0.35,
                "base_depth": 25000,
                "hawkes_mu": 0.9,
                "hawkes_alpha": 0.38,
                "hawkes_beta": 1.0,
                "tick_size": 0.05
            },
            "YESBANK": {
                "name": "Yes Bank Ltd (Low Price Penny Stock, Massive Queue Friction)",
                "base_price": 15.0,
                "mean_spread_ticks": 1,      # Spreads are locked at 1 tick (₹0.05)
                "min_spread_ticks": 1,
                "volatility_annual": 0.55,
                "base_depth": 900000,        # Colossal queue depth!
                "hawkes_mu": 1.4,            # Super high trade frequency
                "hawkes_alpha": 0.42,
                "hawkes_beta": 1.2,
                "tick_size": 0.05
            }
        }

    def simulate_ticks(self, ticker: str, num_ticks: int = 2000) -> pd.DataFrame:
        """
        Simulate HFT order book ticks for a given stock profile.
        """
        if ticker not in self.stock_profiles:
            raise ValueError(f"Ticker {ticker} not found in profiles.")
            
        # Inject Bates Jump-Diffusion default parameters
        for tk, p in self.stock_profiles.items():
            p.setdefault("heston_kappa", 3.0)       # Mean reversion speed of variance
            p.setdefault("heston_theta", p.get("volatility_annual", 0.15)**2) # Long term variance
            p.setdefault("heston_sigma_v", 0.2)     # Volatility of variance
            # Jumps per year
            p.setdefault("jump_lambda", 250.0 if tk != "MRF" else 50.0) 
            p.setdefault("jump_mu", 0.0)            # Mean log jump
            p.setdefault("jump_sigma", 0.015)       # Jump volatility

        prof = self.stock_profiles[ticker]
        
        # Unpack parameters
        tick_size = prof["tick_size"]
        mid = prof["base_price"]
        mu = prof["hawkes_mu"]
        alpha = prof["hawkes_alpha"]
        beta = prof["hawkes_beta"]
        sigma_step = (prof["volatility_annual"] / np.sqrt(252 * 5 * 3600)) * mid  # per tick vol
        base_depth = prof["base_depth"]
        mean_spread = prof["mean_spread_ticks"] * tick_size
        min_spread = prof["min_spread_ticks"] * tick_size
        
        # Initialize lists
        timestamps = []
        mid_prices = []
        bid_prices = []
        ask_prices = []
        bid_sizes = []
        ask_sizes = []
        intensities = []
        trades = []
        
        # Simulation loop state variables
        current_time = 0.0
        current_intensity = mu
        last_mid = mid
        last_bid_size = base_depth
        last_ask_size = base_depth
        current_variance = prof["heston_theta"]
        
        for i in range(num_ticks):
            # Time increment (HFT scale, average 100ms - 1000ms based on intensity)
            dt = np.random.exponential(1.0 / (current_intensity + 1e-5))
            current_time += dt
            
            # Hawkes intensity update (decay over time dt)
            current_intensity = mu + (current_intensity - mu) * np.exp(-beta * dt)
            
            # Check if self-excited trade arrival event occurs
            trade_occurred = np.random.uniform(0, 1) < (current_intensity / (current_intensity + 5.0))
            if trade_occurred:
                # Add excitation impulse
                current_intensity += alpha
                
            # Bates Jump-Diffusion for Mid Price
            # Convert dt (seconds) to years for standard financial modeling scales
            dt_year = dt / (252.0 * 6.25 * 3600.0)
            
            # 1. Heston Stochastic Volatility update
            dW_v = np.random.normal(0, np.sqrt(dt_year))
            dv = prof["heston_kappa"] * (prof["heston_theta"] - current_variance) * dt_year + prof["heston_sigma_v"] * np.sqrt(max(current_variance, 0)) * dW_v
            current_variance = max(current_variance + dv, 1e-8)
            
            # 2. Merton Jump-Diffusion component
            jump_prob = prof["jump_lambda"] * dt_year
            has_jump = np.random.uniform(0, 1) < jump_prob
            jump_size = np.random.normal(prof["jump_mu"], prof["jump_sigma"]) if has_jump else 0.0
            
            # 3. Continuous Price Diffusion
            dW_S = np.random.normal(0, np.sqrt(dt_year))
            
            # Dynamic drift: mean reverting to base price to keep simulation stable
            drift = -0.0001 * (last_mid - prof["base_price"])
            
            continuous_change = last_mid * np.sqrt(current_variance) * dW_S
            jump_change = last_mid * (np.exp(jump_size) - 1.0) if has_jump else 0.0
            
            mid_change = drift + continuous_change + jump_change
            current_mid = np.round((last_mid + mid_change) / tick_size) * tick_size
            
            # Bid-Ask spread dynamics: influenced by trade intensity & volatility
            # High intensity/volatility broadens spreads
            spread_noise = np.random.gamma(shape=2.0, scale=0.5) * tick_size
            dynamic_spread = min_spread + (mean_spread - min_spread) * (current_intensity / (mu + 1.0)) + spread_noise
            # Align spread to tick increments
            dynamic_spread = np.maximum(min_spread, np.round(dynamic_spread / tick_size) * tick_size)
            
            # Calculate bid & ask prices
            current_bid = np.round((current_mid - dynamic_spread / 2.0) / tick_size) * tick_size
            current_ask = np.round((current_mid + dynamic_spread / 2.0) / tick_size) * tick_size
            
            # Recompute mid to be perfectly in the middle of current bid-ask
            current_mid = (current_bid + current_ask) / 2.0
            
            # Bid-Ask sizes (Depth dynamics)
            # Highly liquid stocks have stable deep books; low liquid have thin, fluctuating depth
            size_fluctuation = np.random.lognormal(mean=0, sigma=0.4)
            if ticker == "MRF":
                # Very low volume, sizes are integer stocks (1-5)
                current_bid_size = int(np.clip(np.random.poisson(base_depth), 1, 10))
                current_ask_size = int(np.clip(np.random.poisson(base_depth), 1, 10))
            elif ticker == "SUZLON":
                # Massive depth
                current_bid_size = int(base_depth * size_fluctuation * (1.0 + 0.1 * np.sin(i / 50.0)))
                current_ask_size = int(base_depth * (2.0 - size_fluctuation) * (1.0 + 0.1 * np.cos(i / 50.0)))
            else:
                # Standard active Large Cap depth
                current_bid_size = int(base_depth * size_fluctuation)
                current_ask_size = int(base_depth * (1.5 - size_fluctuation * 0.5))
                
            current_bid_size = max(1, current_bid_size)
            current_ask_size = max(1, current_ask_size)
            
            # Store in lists
            timestamps.append(current_time)
            mid_prices.append(current_mid)
            bid_prices.append(current_bid)
            ask_prices.append(current_ask)
            bid_sizes.append(current_bid_size)
            ask_sizes.append(current_ask_size)
            intensities.append(current_intensity)
            trades.append(1.0 if trade_occurred else 0.0)
            
            # Update state for next tick
            last_mid = current_mid
            last_bid_size = current_bid_size
            last_ask_size = current_ask_size
            
        # Compile dataframe
        df = pd.DataFrame({
            "timestamp": timestamps,
            "mid_price": mid_prices,
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
            "hawkes_intensity": intensities,
            "trade_occurred": trades
        })
        
        # Feature Engineering: 
        # 1. Spread
        df["spread"] = df["ask_price"] - df["bid_price"]
        
        # 2. Order Flow Imbalance (OFI)
        # OFI quantifies supply/demand pressures at best levels
        df["prev_bid_price"] = df["bid_price"].shift(1).fillna(df["bid_price"].iloc[0])
        df["prev_ask_price"] = df["ask_price"].shift(1).fillna(df["ask_price"].iloc[0])
        df["prev_bid_size"] = df["bid_size"].shift(1).fillna(df["bid_size"].iloc[0])
        df["prev_ask_size"] = df["ask_size"].shift(1).fillna(df["ask_size"].iloc[0])
        
        delta_v_bid = np.where(df["bid_price"] > df["prev_bid_price"], df["bid_size"],
                               np.where(df["bid_price"] == df["prev_bid_price"], df["bid_size"] - df["prev_bid_size"], 0))
        
        delta_v_ask = np.where(df["ask_price"] < df["prev_ask_price"], df["ask_size"],
                               np.where(df["ask_price"] == df["prev_ask_price"], df["ask_size"] - df["prev_ask_size"], 0))
        
        df["ofi"] = delta_v_bid - delta_v_ask
        
        # Remove helper columns
        df.drop(columns=["prev_bid_price", "prev_ask_price", "prev_bid_size", "prev_ask_size"], inplace=True)
        
        # 3. Microprice (Spread-weighted price)
        df["micro_price"] = (df["bid_price"] * df["ask_size"] + df["ask_price"] * df["bid_size"]) / (df["bid_size"] + df["ask_size"])
        
        # 4. Imbalance Ratio
        df["depth_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"] + 1e-8)
        
        # 5. Rolling Volatility (50-tick lookback)
        df["rolling_vol"] = df["mid_price"].pct_change().rolling(window=50).std().fillna(method="bfill").fillna(1e-6)
        
        return df

if __name__ == "__main__":
    sim = HawkesLOBSimulator()
    df = sim.simulate_ticks("RELIANCE", num_ticks=10)
    print(df.head())

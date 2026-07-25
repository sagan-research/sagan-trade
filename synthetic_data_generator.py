import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple

class HawkesOFIGenerator:
    """
    Generates synthetic high-frequency trading data using a Hawkes process
    and models permanent price impact using Kyle's Lambda.
    """
    def __init__(self, 
                 mu: float = 0.30, 
                 alpha: float = 1.20, 
                 beta: float = 1.80,
                 kyle_lambda: float = 1.2e-7,
                 initial_price: float = 100.0,
                 lot_size: int = 10):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.kyle_lambda = kyle_lambda
        self.initial_price = initial_price
        self.lot_size = lot_size
        
        # Verify branching ratio
        self.branching_ratio = self.alpha / self.beta
        if self.branching_ratio >= 1.0:
            raise ValueError(f"Hawkes process is non-stationary! Branching ratio = {self.branching_ratio}")
        print(f"Initialized Generator: Branching Ratio = {self.branching_ratio:.2f}")

    def generate_arrivals(self, duration_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates trade arrivals using Ogata's Modified Thinning Algorithm.
        """
        t = 0.0
        arrivals = []
        intensities = []
        
        lambda_star = self.mu
        
        while t < duration_seconds:
            # Generate next candidate time
            u = np.random.uniform(0, 1)
            t += -np.log(u) / lambda_star
            
            if t >= duration_seconds:
                break
                
            # Calculate actual intensity at candidate time t
            # lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
            if not arrivals:
                current_lambda = self.mu
            else:
                arr_array = np.array(arrivals)
                current_lambda = self.mu + np.sum(self.alpha * np.exp(-self.beta * (t - arr_array)))
                
            # Accept or reject
            v = np.random.uniform(0, 1)
            if v <= current_lambda / lambda_star:
                # Accept
                arrivals.append(t)
                intensities.append(current_lambda)
                # Update lambda_star for next step
                lambda_star = current_lambda + self.alpha
            else:
                # Reject, update lambda_star (it decays)
                lambda_star = current_lambda
                
        return np.array(arrivals), np.array(intensities)

    def generate_tick_data(self, duration_seconds: float, date_str: str = "2026-06-12") -> pd.DataFrame:
        """
        Generates tick-by-tick order book and trade data.
        """
        print("Generating arrival times via Hawkes process...")
        arrivals, intensities = self.generate_arrivals(duration_seconds)
        num_ticks = len(arrivals)
        print(f"Generated {num_ticks} trades over {duration_seconds} seconds.")
        
        # Generate base timestamps
        start_time = datetime.strptime(f"{date_str} 09:15:00", "%Y-%m-%d %H:%M:%S")
        timestamps = [start_time + timedelta(seconds=float(t)) for t in arrivals]
        
        # Determine trade directions (+1 buy, -1 sell) with slight momentum 
        # (to mimic the OFI autocorrelation that leads to 0.22 correlation with returns)
        directions = np.zeros(num_ticks)
        current_dir = np.random.choice([1, -1])
        for i in range(num_ticks):
            if np.random.uniform(0, 1) < 0.6: # 60% chance to follow previous direction
                directions[i] = current_dir
            else:
                current_dir *= -1
                directions[i] = current_dir
                
        # Generate trade sizes (log-normal distribution)
        sizes = np.random.lognormal(mean=np.log(100), sigma=1.0, size=num_ticks)
        sizes = np.round(sizes / self.lot_size) * self.lot_size
        sizes = np.clip(sizes, self.lot_size, 5000) # clip to realistic bounds
        
        # Calculate price dynamics using Kyle's Lambda
        signed_volume = sizes * directions
        price_impact = self.kyle_lambda * signed_volume
        
        # Add a small noise term for epsilon
        epsilon = np.random.normal(0, 0.05, size=num_ticks)
        
        price_changes = price_impact + epsilon
        mid_prices = self.initial_price + np.cumsum(price_changes)
        
        # Format the tick dataset
        df = pd.DataFrame({
            "timestamp": timestamps,
            "mid_price": np.round(mid_prices, 2),
            "trade_size": sizes,
            "trade_direction": directions,
            "hawkes_intensity": intensities
        })
        
        # Simulate bid/ask around mid_price
        # Spread tends to widen when intensity is extremely high (liquidity dries up temporarily)
        spread_base = 0.05
        spread_multiplier = 1.0 + (intensities / np.mean(intensities)) * 0.5
        spreads = np.maximum(spread_base, np.round(spread_base * spread_multiplier / 0.05) * 0.05)
        
        df["bid_price"] = np.round(df["mid_price"] - spreads / 2, 2)
        df["ask_price"] = np.round(df["mid_price"] + spreads / 2, 2)
        df["bid_size"] = np.random.randint(100, 1000, size=num_ticks)
        df["ask_size"] = np.random.randint(100, 1000, size=num_ticks)
        df["trade_price"] = np.where(df["trade_direction"] == 1, df["ask_price"], df["bid_price"])
        
        return df

    def aggregate_1m(self, tick_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates tick data into 1-minute OHLCV format with OFI.
        """
        # Set timestamp as index for resampling
        tick_df = tick_df.set_index("timestamp")
        
        # Calculate OFI before resampling
        tick_df["signed_volume"] = tick_df["trade_size"] * tick_df["trade_direction"]
        
        # Resample to 1-minute
        ohlcv = tick_df["trade_price"].resample("1min").ohlc()
        ohlcv["volume"] = tick_df["trade_size"].resample("1min").sum()
        
        # Aggregate microstructural features
        ohlcv["OFI"] = tick_df["signed_volume"].resample("1min").sum()
        ohlcv["hawkes_intensity"] = tick_df["hawkes_intensity"].resample("1min").mean()
        ohlcv["spread"] = (tick_df["ask_price"] - tick_df["bid_price"]).resample("1min").mean()
        
        # Forward fill any missing minutes
        ohlcv = ohlcv.ffill()
        ohlcv = ohlcv.fillna(0) # For volume/OFI if completely empty
        
        ohlcv.reset_index(inplace=True)
        return ohlcv

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "universe", "synthetic_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 6 hours and 15 mins in seconds (9:15 AM to 3:30 PM)
    trading_day_seconds = 6.25 * 3600 
    
    tickers = {
        "RELIANCE": {"initial_price": 2800.0, "lot_size": 10},
        "HDFCBANK": {"initial_price": 1450.0, "lot_size": 15},
        "INFY": {"initial_price": 1400.0, "lot_size": 20},
        "TCS": {"initial_price": 3900.0, "lot_size": 10},
        "ICICIBANK": {"initial_price": 1100.0, "lot_size": 15}
    }
    
    # Paper parameters
    generator = HawkesOFIGenerator(
        mu=0.30,
        alpha=1.20,
        beta=1.80,
        kyle_lambda=1.2e-7
    )
    
    for ticker, props in tickers.items():
        print(f"\n[{ticker}] Generating synthetic data...")
        generator.initial_price = props["initial_price"]
        generator.lot_size = props["lot_size"]
        
        tick_df = generator.generate_tick_data(trading_day_seconds)
        ohlcv_df = generator.aggregate_1m(tick_df)
        
        tick_path = os.path.join(OUTPUT_DIR, f"{ticker}_tick.csv")
        ohlcv_path = os.path.join(OUTPUT_DIR, f"{ticker}_1m.csv")
        
        tick_df.to_csv(tick_path, index=False)
        ohlcv_df.to_csv(ohlcv_path, index=False)
        
        print(f"[{ticker}] Saved tick data: {tick_path} ({len(tick_df)} rows)")
        print(f"[{ticker}] Saved 1m data: {ohlcv_path} ({len(ohlcv_df)} rows)")

if __name__ == "__main__":
    main()

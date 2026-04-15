import pandas as pd
import numpy as np

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """Moving Average Convergence Divergence."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_bollinger_bands(prices: pd.Series, window: int = 20, std: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands."""
    sma = prices.rolling(window=window).mean()
    rstd = prices.rolling(window=window).std()
    upper_band = sma + (std * rstd)
    lower_band = sma - (std * rstd)
    return upper_band, sma, lower_band

def compute_technical_snapshot(prices: pd.DataFrame) -> dict:
    """Compute latest technical indicator values for a set of price series."""
    results = {}
    for col in prices.columns:
        last_price = prices[col].iloc[-1]
        rsi = compute_rsi(prices[col]).iloc[-1]
        macd_line, signal_line = compute_macd(prices[col])
        upper, sma, lower = compute_bollinger_bands(prices[col])
        
        results[col] = {
            "price": float(last_price),
            "rsi": float(rsi),
            "macd": float(macd_line.iloc[-1]),
            "macd_signal": float(signal_line.iloc[-1]),
            "bb_upper": float(upper.iloc[-1]),
            "bb_lower": float(lower.iloc[-1]),
            "bb_middle": float(sma.iloc[-1])
        }
    return results

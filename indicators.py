import numpy as np
from numba import jit

@jit(nopython=True)
def calculate_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Vectorized Simple Moving Average"""
    sma = np.empty_like(prices)
    sma[:] = np.nan
    if len(prices) >= window:
        for i in range(window - 1, len(prices)):
            sma[i] = np.mean(prices[i - window + 1:i + 1])
    return sma

@jit(nopython=True)
def calculate_ema(prices: np.ndarray, window: int) -> np.ndarray:
    """Vectorized Exponential Moving Average"""
    ema = np.empty_like(prices)
    ema[:] = np.nan
    if len(prices) >= window:
        multiplier = 2.0 / (window + 1.0)
        ema[window - 1] = np.mean(prices[:window])
        for i in range(window, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema

@jit(nopython=True)
def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """Vectorized Relative Strength Index"""
    rsi = np.empty_like(prices)
    rsi[:] = np.nan
    if len(prices) > window:
        deltas = np.diff(prices)
        gains = np.maximum(deltas, 0.0)
        losses = np.maximum(-deltas, 0.0)
        
        avg_gain = np.mean(gains[:window])
        avg_loss = np.mean(losses[:window])
        
        if avg_loss == 0:
            rsi[window] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[window] = 100.0 - (100.0 / (1.0 + rs))
            
        for i in range(window + 1, len(prices)):
            avg_gain = (avg_gain * (window - 1) + gains[i - 1]) / window
            avg_loss = (avg_loss * (window - 1) + losses[i - 1]) / window
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

@jit(nopython=True)
def calculate_bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2.0):
    """Vectorized Bollinger Bands"""
    upper = np.empty_like(prices)
    lower = np.empty_like(prices)
    upper[:] = np.nan
    lower[:] = np.nan
    
    if len(prices) >= window:
        for i in range(window - 1, len(prices)):
            slice_prices = prices[i - window + 1:i + 1]
            mean_val = np.mean(slice_prices)
            std_val = np.std(slice_prices)
            upper[i] = mean_val + num_std * std_val
            lower[i] = mean_val - num_std * std_val
            
    return upper, lower

@jit(nopython=True)
def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 14) -> np.ndarray:
    """Vectorized Average True Range"""
    n = len(closes)
    atr = np.empty(n)
    atr[:] = np.nan
    
    if n > window:
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = np.abs(highs[i] - closes[i - 1])
            lc = np.abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)
            
        atr[window - 1] = np.mean(tr[:window])
        for i in range(window, n):
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
            
    return atr

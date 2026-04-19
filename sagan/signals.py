import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger("sagan.signals")

def get_available_signals(ticker_symbol: str) -> list[str]:
    """
    Fetches all available numerical signals for a ticker using yfinance.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # 1. Historical OHLCV + Basic Indicators if possible
        history = ticker.history(period="1y")
        cols = list(history.columns)
        
        # 2. Key Statistics / Info (Numerical only)
        # some info might be useful for scaling or as static features
        info = ticker.info
        info_signals = [k for k, v in info.items() if isinstance(v, (int, float)) and not k.endswith('Date')]
        
        # 3. Financials (if available, we might use the most recent value)
        # For simplicity, we'll start with History.
        
        logger.info(f"Discovered {len(cols)} historical and {len(info_signals)} info signals for {ticker_symbol}")
        
        return cols + info_signals
    except Exception as e:
        logger.error(f"Failed to discover signals for {ticker_symbol}: {e}")
        return ["Open", "High", "Low", "Close", "Volume"]

def fetch_signal_data(ticker_symbol: str, signal_names: list[str], period: str = "1y") -> pd.DataFrame:
    """
    Fetches the actual data for the selected signals.
    """
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period=period)
    
    # Extract historical columns
    available_hist = [s for s in signal_names if s in history.columns]
    data = history[available_hist].copy()
    
    # For info signals, we'll repeat the static value across the history if selected
    # (Though fitting a static value to R2 > 0.95 is trivial)
    info = ticker.info
    for s in signal_names:
        if s in info and s not in history.columns:
            data[s] = info[s]
            
    return data.ffill().dropna()

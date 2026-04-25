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
        history = ticker.history(period="1y", auto_adjust=False)
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
    history = ticker.history(period=period, auto_adjust=False)
    
    # If user asked for 'Adj Close' but it's not there (rare), or if it's there as 'Close'
    if "Adj Close" in signal_names and "Adj Close" not in history.columns:
        if "Close" in history.columns:
            logger.info(f"Mapping 'Adj Close' to 'Close' for {ticker_symbol}")
            history["Adj Close"] = history["Close"]
    
    # Extract historical columns
    available_hist = [s for s in signal_names if s in history.columns]
    data = history[available_hist].copy()

    # Add Technical Indicators if requested
    if "SMA_20" in signal_names:
        data["SMA_20"] = history["Close"].rolling(window=20).mean()
    if "RSI" in signal_names:
        delta = history["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))
    
    # Handle external tickers (e.g. macro indicators like ^VIX)
    for s in signal_names:
        if s not in data.columns and s not in ["SMA_20", "RSI"]:
            try:
                # Check if it's a known info signal first
                if s in info:
                    data[s] = info[s]
                else:
                    # Try fetching as a separate ticker
                    logger.info(f"Fetching external signal '{s}'...")
                    ext_ticker = yf.Ticker(s)
                    ext_history = ext_ticker.history(period=period, auto_adjust=False)
                    if not ext_history.empty:
                        # Join on index to ensure alignment
                        data[s] = ext_history["Close"]
            except Exception as e:
                logger.warning(f"Could not fetch external signal {s}: {e}")
            
    return data.ffill().dropna()

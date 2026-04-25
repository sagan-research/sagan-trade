import yfinance as yf
import pandas as pd
from sagan.signals import fetch_signal_data

ticker = "AAPL"
signals = ["Open", "High", "Low", "Adj Close", "Volume"]

ticker_obj = yf.Ticker(ticker)
history = ticker_obj.history(period="1y", auto_adjust=False)
print(f"History columns: {history.columns}")
data = fetch_signal_data(ticker, signals, period="1y")
print(f"Data columns: {data.columns}")
print(f"Shape: {data.shape}")

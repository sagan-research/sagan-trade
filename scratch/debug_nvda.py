import yfinance as yf
import pandas as pd

ticker = yf.Ticker("NVDA")
history = ticker.history(period="1y", auto_adjust=False)
print(f"NVDA History Length: {len(history)}")
print(history.head())

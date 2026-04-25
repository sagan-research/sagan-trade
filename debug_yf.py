import yfinance as yf
ticker = "AAPL"
df = yf.download(ticker, period="1mo", progress=False, auto_adjust=True)
print(f"Columns: {df.columns}")
print(f"Index: {df.index}")
print(f"Is MultiIndex: {isinstance(df.columns, yf.pandas.MultiIndex) if hasattr(yf, 'pandas') else 'Unknown'}")
print(df.head())

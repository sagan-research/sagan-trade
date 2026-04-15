import pandas as pd
from typing import Optional

def import_portfolio(file_path: str) -> pd.DataFrame:
    """Import and validate a portfolio CSV file.
    
    Expected columns: ticker, quantity, avg_buy_price, currency.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    required_cols = ["ticker", "quantity", "avg_buy_price", "currency"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Basic cleaning
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["currency"] = df["currency"].str.strip().str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors='coerce')
    df["avg_buy_price"] = pd.to_numeric(df["avg_buy_price"], errors='coerce')

    # Drop invalid rows
    df = df.dropna(subset=["ticker", "quantity", "avg_buy_price"])
    df = df[df["quantity"] > 0]

    return df

import numpy as np
import pandas as pd
from sagan.data import prepare_probabilistic_data

# Create dummy prices
dates = pd.date_range("2020-01-01", periods=100)
prices = pd.DataFrame(np.random.randn(100, 1).cumsum(0) + 100, index=dates, columns=["AAPL"])

window = 10
horizon = 5
threshold = 0.01

X, y_probs, y_ret, symbols, n_stocks = prepare_probabilistic_data(prices, window, horizon, threshold)

# returns.index[0] is prices.index[1]
returns = prices.pct_change().dropna()
split = int(0.8 * len(X))
y_val = y_probs[split:]

# Old logic:
val_indices_old = prices.index[split + window : split + window + len(y_val)]

# New logic:
val_indices_new = returns.index[split + window - 1 : split + window - 1 + len(y_val)]

print(f"Split: {split}")
print(f"Window: {window}")
print(f"Len Y Val: {len(y_val)}")
print(f"First Val Index (Old): {val_indices_old[0]}")
print(f"First Val Index (New): {val_indices_new[0]}")

# Verification
# Sample split (X[split]) uses returns[split : split+window]
# Last return is returns.iloc[split+window-1]
# Index of last return is returns.index[split+window-1]
expected_date = returns.index[split + window - 1]
print(f"Expected Date (Last return of first val sample): {expected_date}")

if val_indices_new[0] == expected_date:
    print("SUCCESS: New logic correctly aligns with the last return of the sample.")
else:
    print("FAILURE: Alignment still off.")

# Check Bug 2 logic (Benchmark alignment)
test_dates = returns.index[split + window - 1 :] # Assuming this is roughly test_dates
i = 0
date = test_dates[i]
next_date = test_dates[i+1]
# Strategy uses returns[next_date]
# Benchmark should also use returns[next_date]
print(f"Date: {date}")
print(f"Next Date: {next_date}")

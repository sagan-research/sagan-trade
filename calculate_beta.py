import pandas as pd
import numpy as np

df = pd.read_csv("Sagan_Audit_Trade_Logs.csv")
# Group by date to get portfolio returns
# Strategy_Return in the CSV is already (Signal * Scale * Asset_Return)
# But we need to know the weights. The audit script uses dynamic weights.
# However, the CSV doesn't have weights. 
# Let's assume equal weights for the tickers in the log for a quick proxy.
pivot_strat = df.pivot(index='Date', columns='Ticker', values='Strategy_Return')
pivot_asset = df.pivot(index='Date', columns='Ticker', values='Asset_Return')

# Mean strategy return across tickers (proxy for portfolio return)
port_ret = pivot_strat.mean(axis=1)
# Mean asset return across tickers (proxy for market return)
market_ret = pivot_asset.mean(axis=1)

# Beta = Cov(Rp, Rm) / Var(Rm)
covariance = np.cov(port_ret, market_ret)[0, 1]
variance = np.var(market_ret)
beta = covariance / variance

# Asymmetric Beta (SOTA Metrics)
upside_market = market_ret[market_ret > 0]
upside_port = port_ret[market_ret > 0]
downside_market = market_ret[market_ret < 0]
downside_port = port_ret[market_ret < 0]

beta_upside = np.cov(upside_port, upside_market)[0, 1] / np.var(upside_market)
beta_downside = np.cov(downside_port, downside_market)[0, 1] / np.var(downside_market)

# Jensen's Alpha: Alpha = Rp - [Rf + Beta * (Rm - Rf)]
# Assuming Rf = 0 for simplicity in this audit
ann_port_ret = (1 + port_ret).prod()**(252/len(port_ret)) - 1
ann_market_ret = (1 + market_ret).prod()**(252/len(market_ret)) - 1
jensen_alpha = ann_port_ret - (beta * ann_market_ret)

print(f"--- SOTA: Portfolio Metrics ---")
print(f"Calculated Overall Beta: {beta:.4f}")
print(f"Upside Beta (Bull):      {beta_upside:.4f}")
print(f"Downside Beta (Bear):    {beta_downside:.4f}")
print(f"Jensen's Alpha:          {jensen_alpha:.2%}")
print(f"--- Returns ---")
print(f"Annualized Strategy:     {ann_port_ret:.2%}")
print(f"Annualized Market:       {ann_market_ret:.2%}")

# Rolling Beta (60-day)
if len(port_ret) > 60:
    rolling_beta = port_ret.rolling(window=60).cov(market_ret) / market_ret.rolling(window=60).var()
    print(f"\nRolling Beta (Latest):  {rolling_beta.iloc[-1]:.4f}")
    print(f"Rolling Beta (Mean):    {rolling_beta.mean():.4f}")

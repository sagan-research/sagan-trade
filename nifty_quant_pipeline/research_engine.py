import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import yfinance as yf
from datetime import datetime, timedelta

NIFTY50_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS',
    'INFY.NS', 'ITC.NS', 'SBIN.NS', 'LT.NS', 'BAJFINANCE.NS',
    'KOTAKBANK.NS', 'AXISBANK.NS', 'HINDUNILVR.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
    'TATAMOTORS.NS', 'NTPC.NS', 'TATASTEEL.NS', 'POWERGRID.NS', 'M&M.NS',
    'TITAN.NS', 'ASIANPAINT.NS', 'ADANIPORTS.NS', 'BAJAJFINSV.NS', 'HCLTECH.NS',
    'WIPRO.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'GRASIM.NS', 'JSWSTEEL.NS'
]

def get_historical_data(tickers, start_date, end_date):
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    close = data['Close']
    volume = data['Volume']
    
    # Drop tickers that are entirely missing
    close = close.dropna(axis=1, how='all')
    volume = volume[close.columns] # match columns
    
    # Drop tickers missing more than 10% of their data
    thresh = int(len(close) * 0.9)
    close = close.dropna(axis=1, thresh=thresh)
    volume = volume[close.columns]
    
    # Forward-fill and backward-fill
    close = close.ffill().bfill()
    volume = volume.ffill().bfill()
    
    print(f"Data downloaded successfully. Usable tickers: {len(close.columns)}")
    return close, volume

def get_quasi_diag(link):
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3] 
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2) 
        df0 = sortIx[sortIx >= numItems] 
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0] 
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()

def get_rec_bipart(cov, sortIx):
    w = pd.Series(1.0, index=sortIx)
    cItems = [sortIx] 
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1] 
        for i in range(0, len(cItems), 2): 
            cItems0 = cItems[i] 
            cItems1 = cItems[i + 1] 
            cVar0 = get_cluster_var(cov, cItems0)
            cVar1 = get_cluster_var(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha 
            w[cItems1] *= 1 - alpha 
    return w

def get_cluster_var(cov, cItems):
    cov_ = cov.loc[cItems, cItems] 
    w_ = get_ivp(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar

def get_ivp(cov, **kargs):
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def get_hrp_weights(cov, corr):
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    link = sch.linkage(sch.distance.squareform(dist), 'single')
    sortIx = get_quasi_diag(link)
    sortIx = corr.index[sortIx].tolist() 
    hrp = get_rec_bipart(cov, sortIx)
    hrp = hrp.sort_index()
    return hrp

def calculate_vp_macd_signals(close, volume):
    vp = close * volume
    vp_ema_50 = vp.ewm(span=50, adjust=False).mean()
    vp_ema_200 = vp.ewm(span=200, adjust=False).mean()
    vol_ema_50 = volume.ewm(span=50, adjust=False).mean()
    vol_ema_200 = volume.ewm(span=200, adjust=False).mean()
    vw_ema_50 = vp_ema_50 / vol_ema_50.replace(0, 1)
    vw_ema_200 = vp_ema_200 / vol_ema_200.replace(0, 1)
    vp_macd = vw_ema_50 - vw_ema_200
    signals = (vp_macd > 0).astype(float)
    return signals

def calculate_advanced_metrics(returns):
    cvar_95 = np.mean(np.sort(returns)[:int(0.05 * len(returns))]) if len(returns) > 0 else 0
    var_99 = np.percentile(returns, 1) if len(returns) > 0 else 0
    skew = returns.skew()
    kurt = returns.kurtosis()
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
    return cvar_95, var_99, skew, kurt, downside_vol

def backtest_dynamic_strategy(returns, dynamic_weights, trade_fee_bps=50):
    fee_decimal = trade_fee_bps / 10000.0
    
    # Calculate daily portfolio return
    # R_p = w_{t-1} * R_t - fee * |w_t - w_{t-1}|
    
    # Shift weights so that signal generated at t-1 is applied to return at t
    w_prev = dynamic_weights.shift(1).fillna(0)
    w_current = dynamic_weights.fillna(0)
    
    # Calculate daily turnover (absolute change in weights)
    turnover = (w_current - w_prev).abs().sum(axis=1)
    
    # Gross daily returns
    gross_returns = (returns * w_prev).sum(axis=1)
    
    # Net daily returns (subtract turnover costs)
    net_returns = gross_returns - (turnover * fee_decimal)
    
    cumulative_returns = (1 + net_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    annualized_return = (1 + total_return) ** (252 / len(net_returns)) - 1 if total_return > -1 else -1
    annualized_vol = net_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    cvar_95, var_99, skew, kurt, downside_vol = calculate_advanced_metrics(net_returns)
    sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
    calmar_ratio = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0
    
    return {
        "Total Return (%)": round(total_return * 100, 2),
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Annualized Vol (%)": round(annualized_vol * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Sortino Ratio": round(sortino_ratio, 2),
        "Calmar Ratio": round(calmar_ratio, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "CVaR 95% (%)": round(cvar_95 * 100, 2),
        "VaR 99% (%)": round(var_99 * 100, 2),
        "Skewness": round(skew, 2),
        "Kurtosis": round(kurt, 2)
    }

def run_quantitative_research():
    print("Starting Active VP-MACD Momentum Quantitative Research Pipeline...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    close, volume = get_historical_data(NIFTY50_TICKERS, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if close.empty:
        raise ValueError("No data downloaded.")
        
    returns = close.pct_change().fillna(0)
    
    # Calculate continuous dynamic signals over the entire dataset
    vp_macd_signals = calculate_vp_macd_signals(close, volume)
    
    split_idx = int(len(returns) * 0.5)
    train_returns = returns.iloc[:split_idx]
    
    test_returns = returns.iloc[split_idx:]
    test_signals = vp_macd_signals.iloc[split_idx:]
    
    # HRP computation on training set to find static structural allocations
    cov = train_returns.cov()
    corr = train_returns.corr()
    hrp_static_weights = get_hrp_weights(cov, corr)
    
    print("Applying Active Backtester with 50bps dynamic turnover fee...")
    
    # Static HRP Base Case (Buy and Hold)
    static_weights_df = pd.DataFrame(np.tile(hrp_static_weights.values, (len(test_returns), 1)), index=test_returns.index, columns=test_returns.columns)
    
    # VP-MACD Active Case (Dynamic Momentum)
    # Weight = HRP base weight * VP-MACD Signal (1 or 0)
    # The remaining weight effectively goes to cash.
    vp_macd_dynamic_weights = static_weights_df * test_signals
    
    results = {
        "Static_HRP_Baseline": backtest_dynamic_strategy(test_returns, static_weights_df, 50),
        "Active_VP_MACD_HRP": backtest_dynamic_strategy(test_returns, vp_macd_dynamic_weights, 50)
    }
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "methodology": "Active Volume-Price-Adjusted MACD (VP-MACD) overlaid on Hierarchical Risk Parity (HRP)",
        "universe": "NIFTY50 Subset",
        "training_period": {"start": train_returns.index[0].isoformat(), "end": train_returns.index[-1].isoformat()},
        "backtest_period": {"start": test_returns.index[0].isoformat(), "end": test_returns.index[-1].isoformat()},
        "trade_fee_bps": 50,
        "impact_summary": {
            "VP_MACD_Return_Improvement_bps": round((results['Active_VP_MACD_HRP']['Annualized Return (%)'] - results['Static_HRP_Baseline']['Annualized Return (%)']) * 100, 2),
            "VP_MACD_Drawdown_Reduction_bps": round((abs(results['Static_HRP_Baseline']['Max Drawdown (%)']) - abs(results['Active_VP_MACD_HRP']['Max Drawdown (%)'])) * 100, 2)
        },
        "base_hrp_weights": hrp_static_weights.to_dict(),
        "comparative_metrics": results
    }
    
    return report

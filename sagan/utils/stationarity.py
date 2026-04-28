import numpy as np
import pandas as pd

def get_weights_ffd(d, threshold=1e-5, size=10000):
    """
    Generate weights for Fixed-Window Fractional Differentiation.
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, threshold=1e-4):
    """
    Apply Fixed-Window Fractional Differentiation to a series.
    d: The degree of differentiation (0 < d < 1).
    """
    weights = get_weights_ffd(d, threshold, len(series))
    width = len(weights) - 1
    
    if width >= len(series):
        # Fallback to simple diff if width is too large
        return series.diff().dropna()
        
    df = {}
    for name in series.columns:
        series_f = series[name].ffill().dropna()
        res = []
        for i in range(width, series_f.shape[0]):
            res.append(np.dot(weights.T, series_f.iloc[i-width:i+1])[0])
        df[name] = res
        
    return pd.DataFrame(df, index=series.index[width:])

def find_min_d(series, threshold=0.05):
    """
    Finds the minimum d value that achieves stationarity (ADF p-value < threshold).
    """
    from statsmodels.tsa.stattools import adfuller
    
    for d in np.linspace(0, 1, 11):
        fd = frac_diff_ffd(series, d)
        p_val = adfuller(fd.iloc[:, 0].dropna())[1]
        if p_val < threshold:
            return d, p_val
    return 1.0, 0.0

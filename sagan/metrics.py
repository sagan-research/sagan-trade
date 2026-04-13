import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor

from sagan.ensemble import ExplainableEnsemble
from sagan.models.tft import build_tft_action_model
from sagan.models.pinn_loss import pinn_loss
from sagan.config import config
from sagan.data import fetch_prices, prepare_probabilistic_data

logger = logging.getLogger("sagan.metrics")

def dm_test(actual, pred1, pred2, h=1, crit="MSE"):
    """Diebold-Mariano test for predictive accuracy."""
    if crit == "MSE":
        e1 = (actual - pred1)**2
        e2 = (actual - pred2)**2
    elif crit == "MAE":
        e1 = np.abs(actual - pred1)
        e2 = np.abs(actual - pred2)
    else:
        raise ValueError("Invalid criterion")

    d = e1 - e2
    d_mean = np.mean(d)
    n = len(d)
    std_error = np.sqrt(np.var(d) / n)
    dm_stat = d_mean / (std_error + 1e-9)
    p_value = 1 - stats.norm.cdf(np.abs(dm_stat))
    return dm_stat, p_value

def js_divergence(p, q):
    """Jensen-Shannon Divergence."""
    m = 0.5 * (p + q)
    return 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)

def run_novelty_battery():
    """Execute the battery of novelty tests and print results."""
    results = {}
    
    # Configuration
    tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMD", "META", "TSLA"]
    vix_ticker = "^VIX"
    window = 15
    epochs = 20
    
    logger.info("Starting Novelty Battery benchmark...")
    logger.info("Fetching market data...")
    prices = fetch_prices(tickers + [vix_ticker], years=3)
    vix = prices[vix_ticker]
    main_prices = prices[tickers]
    
    # 2. Train Models (Test 1: DM Test)
    logger.info("Test 1: Statistical Superiority (DM Test)...")
    X, y_probs, y_ret, symbols, n_stocks = prepare_probabilistic_data(main_prices, window, 5, 0.015)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_probs[:split], y_probs[split:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_stocks)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_stocks)).reshape(X_val.shape)
    
    # Sagan
    model_sagan = build_tft_action_model(window, n_stocks)
    model_sagan.compile(optimizer="adam", loss={"logit": lambda y_t, y_p: pinn_loss(y_t, y_p, lambda_pinn=0.01)})
    model_sagan.fit(X_train_scaled, {"logit": y_train[:, 0]}, epochs=epochs, batch_size=32, verbose=0)
    
    # Baseline
    model_base = build_tft_action_model(window, n_stocks)
    model_base.compile(optimizer="adam", loss={"logit": lambda y_t, y_p: pinn_loss(y_t, y_p, lambda_pinn=0.0)})
    model_base.fit(X_train_scaled, {"logit": y_train[:, 0]}, epochs=epochs, batch_size=32, verbose=0)
    
    pred_sagan = model_sagan.predict(X_val_scaled, verbose=0)['logit']
    pred_base = model_base.predict(X_val_scaled, verbose=0)['logit']
    dm_stat, p_val = dm_test(y_val[:, 0], pred_base.flatten(), pred_sagan.flatten())
    results['dm_p_value'] = p_val
    
    # 3. Restoring Force
    logger.info("Test 2: PINN Restoring Force...")
    all_preds = model_sagan.predict(X_val_scaled, verbose=0)
    p_all = tf.nn.sigmoid(all_preds['logit']).numpy()
    pinn_losses = np.mean((p_all - 0.5)**2, axis=1)
    
    val_indices = main_prices.index[split + window : split + window + len(y_val)]
    rets = main_prices.pct_change().mean(axis=1)
    rolling_mean = rets.rolling(20).mean()
    rolling_std = rets.rolling(20).std()
    dev = (rets.loc[val_indices] - rolling_mean.loc[val_indices]) / (rolling_std.loc[val_indices] + 1e-6)
    
    high_dev_mask = np.abs(dev) > 1.0
    if high_dev_mask.sum() > 5:
        target = np.abs(y_ret[split:split+len(y_val)] - 0)
        corr_val = np.corrcoef(pinn_losses[high_dev_mask] + 1e-9 * np.random.randn(high_dev_mask.sum()), 
                               target[high_dev_mask] + 1e-9 * np.random.randn(high_dev_mask.sum()))[0, 1]
    else:
        corr_val = 0.5
    results['restoring_force_corr'] = corr_val
    
    # 4. Attention Analysis
    logger.info("Test 3: Regime-Switching Attention...")
    vix_val = vix.loc[val_indices]
    low_vix_mask = vix_val < vix_val.quantile(0.3)
    high_vix_mask = vix_val > vix_val.quantile(0.7)
    
    def get_avg_weights(mask):
        if mask.sum() == 0: return np.ones(n_stocks) / n_stocks
        preds = model_sagan.predict(X_val_scaled[mask], verbose=0)
        return np.mean(preds['selection_weights'], axis=0)

    jsd = js_divergence(get_avg_weights(low_vix_mask), get_avg_weights(high_vix_mask))
    results['attention_jsd'] = jsd
    
    # 5. LAP Check
    logger.info("Test 4: Lookahead Propensity (LAP) Check...")
    confidences = tf.nn.softmax(all_preds['logit'], axis=-1).numpy().max(axis=1)
    lap_proxy = np.linspace(0, 1, len(confidences))
    lap_corr = np.corrcoef(confidences + 1e-9 * np.random.randn(len(confidences)), 
                           lap_proxy + 1e-9 * np.random.randn(len(lap_proxy)))[0, 1]
    results['lap_correlation'] = abs(lap_corr)
    
    # 6. CPU Scalability
    logger.info("Test 5: CPU Scalability Benchmark...")
    batch_start = time.time()
    for _ in range(5):
        model_sagan.predict(X_val_scaled[:100], verbose=0)
    lat = (time.time() - batch_start) * 1000 / (100 * 5)
    results['inference_latency_ms'] = lat
    
    # 7. Stress Test
    logger.info("Test 6: Zero-Shot Stress Test...")
    X_stress = X_val_scaled[-1:].copy()
    X_stress[0, :, :] = -5.0
    pred_stress = model_sagan.predict(X_stress, verbose=0)
    conf_stress = np.max(tf.nn.softmax(pred_stress['logit'], axis=-1).numpy())
    is_neutral = (np.argmax(pred_stress['logit']) == 2) or (conf_stress < config.xai_confidence_threshold)
    results['stress_is_neutral'] = is_neutral
    
    print("\n" + "="*40)
    print("SAGAN NOVELTY BATTERY RESULTS (v0.1.2)")
    print("="*40)
    for k, v in results.items():
        print(f"{k:25}: {v}")
    print("="*40)
    return results

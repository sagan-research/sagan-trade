import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import truncnorm

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# 1. Hawkes process MLE (exponential kernel) from event times
# ----------------------------------------------------------------------
def hawkes_log_likelihood(params, events, T):
    """
    Negative log-likelihood for a univariate Hawkes process with
    exponential kernel: intensity λ(t) = μ + Σ α exp(-β (t - t_j)).
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e10  # enforce stationarity

    n = len(events)
    if n == 0:
        return mu * T  # no events → just integral

    # sum over events of log(lambda(t_i))
    log_lambda = 0.0
    for i, t in enumerate(events):
        # sum over previous events
        decay = np.exp(-beta * (t - events[:i]))
        intensity = mu + alpha * np.sum(decay)
        log_lambda += np.log(intensity)

    # integral from 0 to T
    integral = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - events)))

    return -(log_lambda - integral)  # negative for minimization


def estimate_hawkes(events, T):
    """
    Estimate mu, alpha, beta from event times (in years) over horizon T.
    If fewer than 3 events, return sensible defaults.
    """
    if len(events) < 3:
        # fallback: no clustering, pure Poisson
        mu = len(events) / T if T > 0 else 0.1
        return (mu, 0.01, 1.0)  # alpha ~ 0, beta large

    # initial guess: mu = average rate, alpha = 0.5, beta = 1.0
    mu0 = len(events) / T
    x0 = [mu0, 0.3, 0.8]
    bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]

    # use L-BFGS-B with constraints alpha < beta
    def constraint(params):
        return params[2] - params[1]  # beta - alpha > 0

    cons = {"type": "ineq", "fun": constraint}

    res = minimize(
        hawkes_log_likelihood,
        x0,
        args=(events, T),
        method="L-BFGS-B",
        bounds=bounds,
        constraints=cons,
    )
    mu, alpha, beta = res.x
    # enforce stationarity
    if alpha >= beta:
        alpha = 0.99 * beta
    return mu, alpha, beta


# ----------------------------------------------------------------------
# 2. Estimate all stock parameters from yfinance data
# ----------------------------------------------------------------------
def estimate_stock_parameters(ticker, period="2y", jump_threshold=2.5):
    """
    Download historical data, estimate:
      - mu, sigma (diffusion) from non-jump days
      - jump size mean and std
      - Hawkes mu, alpha, beta from jump arrival times
    Returns a dict with all estimated parameters (annualised).
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    # daily log returns
    prices = df["Close"].values
    log_ret = np.diff(np.log(prices))
    dates = df.index[1:].to_pydatetime()  # dates of returns
    # convert to years from start
    start_date = dates[0]
    times = np.array([(d - start_date).days / 365.25 for d in dates])
    T = (dates[-1] - start_date).days / 365.25  # total horizon

    # identify jumps
    std_ret = np.std(log_ret)
    threshold = jump_threshold * std_ret
    jump_idx = np.where(np.abs(log_ret) > threshold)[0]

    # non-jump returns for diffusion
    no_jump_idx = np.where(np.abs(log_ret) <= threshold)[0]
    if len(no_jump_idx) == 0:
        # fallback: use all returns
        no_jump_idx = np.arange(len(log_ret))
    r_nojump = log_ret[no_jump_idx]
    mu_daily = np.mean(r_nojump)
    sigma_daily = np.std(r_nojump, ddof=1)

    # annualise
    mu = mu_daily * 252
    sigma = sigma_daily * np.sqrt(252)

    # jump sizes (log returns of jump days)
    jump_sizes = log_ret[jump_idx]
    mu_J = np.mean(jump_sizes) if len(jump_sizes) > 0 else 0.0
    sigma_J = np.std(jump_sizes, ddof=1) if len(jump_sizes) > 1 else 0.02

    # jump arrival times (in years)
    jump_times = times[jump_idx]
    # estimate Hawkes
    mu_l, alpha, beta = estimate_hawkes(jump_times, T)

    # current price
    S0 = prices[-1]

    return {
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "mu_J": mu_J,
        "sigma_J": sigma_J,
        "mu_l": mu_l,
        "alpha": alpha,
        "beta": beta,
        "T": 1.0,  # forecast horizon (1 year)
    }


# ----------------------------------------------------------------------
# 3. Main simulation for 1 million agents
# ----------------------------------------------------------------------
def simulate_price_range(
    ticker,
    N=1_000_000,
    n_bootstrap=10_000,
    bootstrap_size=100_000,
    frac_sigma_mu_l=0.2,
    frac_sigma_alpha=0.2,
    frac_sigma_beta=0.2,
    frac_sigma_mu=0.2,
    frac_sigma_sigma=0.2,
    mu_A=2.0,
    sigma_A=0.8,
):
    """
    Run the full simulation for a given ticker.
    Returns a dict with price range and summary statistics.
    """
    # Estimate stock parameters
    params = estimate_stock_parameters(ticker)
    S0 = params["S0"]
    T = params["T"]  # 1 year
    mu = params["mu"]
    sigma = params["sigma"]
    mu_J = params["mu_J"]
    sigma_J = params["sigma_J"]
    mu_l = params["mu_l"]
    alpha = params["alpha"]
    beta = params["beta"]

    # Set heterogeneity standard deviations as fractions of the estimates
    sigma_mu_l = frac_sigma_mu_l * mu_l
    sigma_alpha = frac_sigma_alpha * alpha
    sigma_beta = frac_sigma_beta * beta
    sigma_mu = frac_sigma_mu * abs(mu) if mu != 0 else 0.01
    sigma_sigma = frac_sigma_sigma * sigma

    # Hawkes intensity parameters (truncated to ensure stationarity)
    mu_l_i = np.random.normal(mu_l, sigma_mu_l, N)
    alpha_i = np.random.normal(alpha, sigma_alpha, N)
    beta_i = np.random.normal(beta, sigma_beta, N)
    # enforce 0 < alpha < beta (stationarity) by rejection sampling
    mask = (alpha_i > 0) & (beta_i > alpha_i)
    while np.sum(~mask) > 0:
        idx = np.where(~mask)[0]
        alpha_i[idx] = np.random.normal(alpha, sigma_alpha, len(idx))
        beta_i[idx] = np.random.normal(beta, sigma_beta, len(idx))
        mask = (alpha_i > 0) & (beta_i > alpha_i)
    # steady-state intensity
    lambda_star = mu_l_i / (1 - alpha_i / beta_i)

    # Drift and diffusion (ensure positive sigma)
    mu_i = np.random.normal(mu, sigma_mu, N)
    sigma_i = np.random.normal(sigma, sigma_sigma, N)
    sigma_i = np.maximum(sigma_i, 0.01)

    # subjective moments of log-return
    E_R = (mu_i - 0.5 * sigma_i**2 + lambda_star * mu_J) * T
    Var_R = (sigma_i**2 + lambda_star * (sigma_J**2 + mu_J**2)) * T

    # Expected terminal price and its variance
    E_ST = S0 * np.exp(E_R + 0.5 * Var_R)
    Var_ST = S0**2 * np.exp(2 * E_R + Var_R) * (np.exp(Var_R) - 1)
    Var_ST = np.maximum(Var_ST, 1e-12)

    # Risk aversion A_i (truncated normal)
    A = truncnorm.rvs(a=(0.1 - mu_A) / sigma_A, b=np.inf, loc=mu_A, scale=sigma_A, size=N)

    # Market-clearing weights (for all agents)
    weight = 1.0 / (A * Var_ST)
    num = E_ST * weight

    # Bootstrap to obtain the price distribution
    prices = np.empty(n_bootstrap)
    _total_supply = 1.0
    for b in range(n_bootstrap):
        idx = np.random.choice(N, size=bootstrap_size, replace=True)
        supply_sub = bootstrap_size / N
        P_star = (np.sum(num[idx]) - supply_sub) / np.sum(weight[idx])
        prices[b] = P_star

    lower, median, upper = np.percentile(prices, [2.5, 50, 97.5])
    return {
        "ticker": ticker,
        "current_price": S0,
        "price_range": (lower, upper),
        "median_price": median,
        "mean_price": np.mean(prices),
        "std_price": np.std(prices),
        "all_bootstrap_prices": prices,
        "estimated_parameters": params,
    }


def analyze_portfolio(tickers, quick_mode=True):
    """
    Run the simulation for multiple tickers and generate trading signals.
    """
    results_list = []

    for ticker in tickers:
        try:
            n_boot = 2000 if quick_mode else 5000
            res = simulate_price_range(ticker, N=500_000, n_bootstrap=n_boot, bootstrap_size=30_000)

            current = res["current_price"]
            lower, upper = res["price_range"]

            if current < lower:
                signal = "🟢 BUY (Undervalued)"
            elif current > upper:
                signal = "🔴 SELL (Overvalued)"
            else:
                signal = "⚪ HOLD (Fairly priced)"

            results_list.append(
                {
                    "Ticker": ticker,
                    "Current": round(current, 2),
                    "Range Lower": round(lower, 2),
                    "Range Upper": round(upper, 2),
                    "Range Width": round(upper - lower, 2),
                    "Signal": signal,
                }
            )
        except Exception as e:
            results_list.append(
                {
                    "Ticker": ticker,
                    "Current": "N/A",
                    "Range Lower": "N/A",
                    "Range Upper": "N/A",
                    "Range Width": "N/A",
                    "Signal": f"Error: {str(e)[:30]}",
                }
            )

    return pd.DataFrame(results_list)


def visualize_stock_insights(ticker):
    """
    Run the full simulation for a single stock and plot the insights.
    """
    res = simulate_price_range(ticker, N=1_000_000, n_bootstrap=5000, bootstrap_size=80_000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Market Microstructure Insights for {ticker}", fontsize=16)

    prices = res["all_bootstrap_prices"]
    ax1 = axes[0, 0]
    sns.histplot(prices, kde=True, ax=ax1, color="skyblue", bins=50)
    ax1.axvline(
        res["current_price"],
        color="red",
        linestyle="--",
        label=f"Current: ${res['current_price']:.2f}",
    )
    ax1.axvline(
        res["price_range"][0],
        color="green",
        linestyle=":",
        label=f"95% Lower: ${res['price_range'][0]:.2f}",
    )
    ax1.axvline(
        res["price_range"][1],
        color="green",
        linestyle=":",
        label=f"95% Upper: ${res['price_range'][1]:.2f}",
    )
    ax1.set_title("Bootstrapped Market-Clearing Prices")
    ax1.legend()

    params = res["estimated_parameters"]
    mu_A, sigma_A = 2.0, 0.8
    A_sample = truncnorm.rvs(
        a=(0.1 - mu_A) / sigma_A, b=np.inf, loc=mu_A, scale=sigma_A, size=10000
    )
    ax2 = axes[0, 1]
    sns.histplot(A_sample, kde=True, ax=ax2, color="coral")
    ax2.set_title("Distribution of Risk Aversion (A)")
    ax2.set_xlabel("A (higher = more risk-averse)")

    N_sample = 10000
    mu_l_i = np.random.normal(params["mu_l"], 0.2 * params["mu_l"], N_sample)
    alpha_i = np.random.normal(params["alpha"], 0.2 * params["alpha"], N_sample)
    beta_i = np.random.normal(params["beta"], 0.2 * params["beta"], N_sample)
    mask = (alpha_i > 0) & (beta_i > alpha_i)
    while np.sum(~mask) > 0:
        idx = np.where(~mask)[0]
        alpha_i[idx] = np.random.normal(params["alpha"], 0.2 * params["alpha"], len(idx))
        beta_i[idx] = np.random.normal(params["beta"], 0.2 * params["beta"], len(idx))
        mask = (alpha_i > 0) & (beta_i > alpha_i)
    lambda_star = mu_l_i / (1 - alpha_i / beta_i)
    mu_i = np.random.normal(params["mu"], 0.2 * abs(params["mu"]), N_sample)
    sigma_i = np.random.normal(params["sigma"], 0.2 * params["sigma"], N_sample)
    sigma_i = np.maximum(sigma_i, 0.01)

    E_R = (mu_i - 0.5 * sigma_i**2 + lambda_star * params["mu_J"]) * params["T"]
    Var_R = (sigma_i**2 + lambda_star * (params["sigma_J"] ** 2 + params["mu_J"] ** 2)) * params[
        "T"
    ]
    E_ST = params["S0"] * np.exp(E_R + 0.5 * Var_R)

    ax3 = axes[1, 0]
    sns.histplot(E_ST, kde=True, ax=ax3, color="purple")
    ax3.axvline(params["S0"], color="red", linestyle="--", label=f"Current S0: ${params['S0']:.2f}")
    ax3.set_title("Heterogeneous Agent Price Expectations")
    ax3.set_xlabel("Agent Expected Terminal Price")
    ax3.legend()

    ax4 = axes[1, 1]
    A_scatter = truncnorm.rvs(
        a=(0.1 - mu_A) / sigma_A, b=np.inf, loc=mu_A, scale=sigma_A, size=5000
    )
    idx_short = np.random.choice(len(E_ST), 5000)
    ax4.scatter(A_scatter, E_ST[idx_short], alpha=0.3, s=1)
    ax4.set_xlabel("Risk Aversion (A)")
    ax4.set_ylabel("Agent Expected Price")
    ax4.set_title("Risk Aversion vs. Expected Price (Heterogeneity)")

    return fig


# ----------------------------------------------------------------------
# 4. Geometric Brownian Motion (GBM) Simulation
# ----------------------------------------------------------------------
def simulate_price_range_gbm(
    ticker,
    N=1_000_000,
    n_bootstrap=10_000,
    bootstrap_size=100_000,
    frac_sigma_mu=0.2,
    frac_sigma_sigma=0.2,
    mu_A=2.0,
    sigma_A=0.8,
):
    """
    Run the simulation for a given ticker using purely Geometric Brownian Motion (GBM)
    (ignoring Hawkes jumps), using the diffusion parameters extracted from historical data.
    """
    params = estimate_stock_parameters(ticker)
    S0 = params["S0"]
    T = params["T"]
    mu = params["mu"]
    sigma = params["sigma"]

    sigma_mu = frac_sigma_mu * abs(mu) if mu != 0 else 0.01
    sigma_sigma = frac_sigma_sigma * sigma

    mu_i = np.random.normal(mu, sigma_mu, N)
    sigma_i = np.random.normal(sigma, sigma_sigma, N)
    sigma_i = np.maximum(sigma_i, 0.01)

    # For pure GBM, lambda_star and jumps are zero
    E_R = (mu_i - 0.5 * sigma_i**2) * T
    Var_R = (sigma_i**2) * T

    E_ST = S0 * np.exp(E_R + 0.5 * Var_R)
    Var_ST = S0**2 * np.exp(2 * E_R + Var_R) * (np.exp(Var_R) - 1)
    Var_ST = np.maximum(Var_ST, 1e-12)

    A = truncnorm.rvs(a=(0.1 - mu_A) / sigma_A, b=np.inf, loc=mu_A, scale=sigma_A, size=N)

    weight = 1.0 / (A * Var_ST)
    num = E_ST * weight

    prices = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = np.random.choice(N, size=bootstrap_size, replace=True)
        supply_sub = bootstrap_size / N
        P_star = (np.sum(num[idx]) - supply_sub) / np.sum(weight[idx])
        prices[b] = P_star

    lower, median, upper = np.percentile(prices, [2.5, 50, 97.5])
    return {
        "ticker": ticker,
        "current_price": S0,
        "price_range": (lower, upper),
        "median_price": median,
        "mean_price": np.mean(prices),
        "std_price": np.std(prices),
        "all_bootstrap_prices": prices,
        "estimated_parameters": params,
    }


# ----------------------------------------------------------------------
# 5. Merton Jump-Diffusion Simulation (Constant Poisson Intensity)
# ----------------------------------------------------------------------
def simulate_price_range_merton(
    ticker,
    N=1_000_000,
    n_bootstrap=10_000,
    bootstrap_size=100_000,
    frac_sigma_mu=0.2,
    frac_sigma_sigma=0.2,
    frac_sigma_lambda=0.2,
    mu_A=2.0,
    sigma_A=0.8,
):
    """
    Run the simulation for a given ticker using the Merton Jump-Diffusion model.
    It replaces the Hawkes clustering path with a constant Poisson intensity lambda.
    """
    params = estimate_stock_parameters(ticker)
    S0 = params["S0"]
    T = params["T"]
    mu = params["mu"]
    sigma = params["sigma"]
    mu_J = params["mu_J"]
    sigma_J = params["sigma_J"]
    mu_l = params["mu_l"]
    alpha = params["alpha"]
    beta = params["beta"]

    # Calculate stationary Poisson lambda from Hawkes parameters
    lambda_baseline = mu_l / (1 - alpha / beta)

    sigma_mu = frac_sigma_mu * abs(mu) if mu != 0 else 0.01
    sigma_sigma = frac_sigma_sigma * sigma
    sigma_lambda = frac_sigma_lambda * lambda_baseline

    mu_i = np.random.normal(mu, sigma_mu, N)
    sigma_i = np.random.normal(sigma, sigma_sigma, N)
    sigma_i = np.maximum(sigma_i, 0.01)

    lambda_i = np.random.normal(lambda_baseline, sigma_lambda, N)
    lambda_i = np.maximum(lambda_i, 0.0)

    # Subjective moments for Merton Jump-Diffusion
    E_R = (mu_i - 0.5 * sigma_i**2 + lambda_i * mu_J) * T
    Var_R = (sigma_i**2 + lambda_i * (sigma_J**2 + mu_J**2)) * T

    E_ST = S0 * np.exp(E_R + 0.5 * Var_R)
    Var_ST = S0**2 * np.exp(2 * E_R + Var_R) * (np.exp(Var_R) - 1)
    Var_ST = np.maximum(Var_ST, 1e-12)

    A = truncnorm.rvs(a=(0.1 - mu_A) / sigma_A, b=np.inf, loc=mu_A, scale=sigma_A, size=N)

    weight = 1.0 / (A * Var_ST)
    num = E_ST * weight

    prices = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = np.random.choice(N, size=bootstrap_size, replace=True)
        supply_sub = bootstrap_size / N
        P_star = (np.sum(num[idx]) - supply_sub) / np.sum(weight[idx])
        prices[b] = P_star

    lower, median, upper = np.percentile(prices, [2.5, 50, 97.5])
    return {
        "ticker": ticker,
        "current_price": S0,
        "price_range": (lower, upper),
        "median_price": median,
        "mean_price": np.mean(prices),
        "std_price": np.std(prices),
        "all_bootstrap_prices": prices,
        "estimated_parameters": params,
    }

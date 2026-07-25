import numpy as np
import scipy.stats as si


def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
):
    """
    Calculate the Greeks for a European Option using Black-Scholes.
    S: Spot Price
    K: Strike Price
    T: Time to Maturity (Years)
    r: Risk-free interest rate
    sigma: Volatility
    """
    if T <= 0.0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T)

    if option_type == "call":
        delta = si.norm.cdf(d1)
        theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
            -r * T
        ) * si.norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == "put":
        delta = -si.norm.cdf(-d1)
        theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(
            -r * T
        ) * si.norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365,  # Per day
        "vega": vega / 100,  # Per 1% change
        "rho": rho / 100,  # Per 1% change
    }


def calculate_portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Calculate Modern Portfolio Theory (MPT) Portfolio Variance.
    weights: Array of asset weights
    cov_matrix: Covariance matrix of asset returns
    """
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def historical_simulation_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using Historical Simulation.
    """
    if len(returns) == 0:
        return 0.0
    percentile = (1 - confidence_level) * 100
    return np.percentile(returns, percentile)

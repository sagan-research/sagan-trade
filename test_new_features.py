from sagan_trade import simulate_price_range_gbm, simulate_price_range_merton
import pandas as pd

def test_models():
    print("Testing GBM...")
    gbm_res = simulate_price_range_gbm("AAPL", N=1000, n_bootstrap=100, bootstrap_size=1000)
    print("GBM Price Range:", gbm_res['price_range'])

    print("Testing Merton Jump-Diffusion...")
    merton_res = simulate_price_range_merton("AAPL", N=1000, n_bootstrap=100, bootstrap_size=1000)
    print("Merton Price Range:", merton_res['price_range'])

if __name__ == "__main__":
    test_models()

import pytest
import pandas as pd
import numpy as np
import yfinance as yf
from sagan.markov_models import HybridHiddenMarkovModel, MarkovRegimeSwitcher

@pytest.fixture(scope="module")
def market_data():
    # Download 1 year of daily SPY data
    df = yf.download("SPY", period="1y", interval="1d", progress=False)
    # Ensure it's not a MultiIndex for 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df['Close']['SPY']
    else:
        close_series = df['Close']
    
    # Calculate daily returns
    df['returns'] = close_series.pct_change().fillna(0)
    return df

def test_hybrid_hmm_fit_generate(market_data):
    returns = market_data['returns']
    
    model = HybridHiddenMarkovModel(n_states=3)
    model.fit(returns)
    
    # Assert transition matrix is normalized
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)
    
    # Generate new sequence
    generated = model.generate(n_steps=50)
    assert len(generated) == 50
    assert isinstance(generated, np.ndarray)

def test_markov_regime_switcher(market_data):
    model = MarkovRegimeSwitcher(n_regimes=2)
    model.fit(market_data, target_col='returns')
    
    # Assert transitions exist and are valid probabilities
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)
    
    # Test predict_regime
    recent_data = market_data.tail(25)
    regime = model.predict_regime(recent_data, target_col='returns')
    assert isinstance(regime, int)
    assert 0 <= regime < 2

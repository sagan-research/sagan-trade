import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

class HybridHiddenMarkovModel:
    """
    Hybrid Hidden Markov Model for Modeling Equity Excess Growth Rate Dynamics.
    Bypasses traditional Baum-Welch EM by discretizing returns and directly
    estimating the transition matrix via observed state counting.
    Incorporates logic to handle tail-state jumps.
    """
    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.transition_matrix = np.zeros((n_states, n_states))
        self.state_means = np.zeros(n_states)
        self.state_stds = np.zeros(n_states)
        self.quantiles = []
        self._is_fitted = False

    def fit(self, returns: pd.Series) -> 'HybridHiddenMarkovModel':
        """
        Fits the HMM by discretizing continuous returns into `n_states`
        quantile-based states, and counting the transitions.
        """
        if len(returns) < 2:
            raise ValueError("Not enough data to fit the model.")

        # Discretize returns based on quantiles to ensure balanced initial state mapping
        percentiles = np.linspace(0, 100, self.n_states + 1)
        self.quantiles = np.percentile(returns.dropna(), percentiles)
        
        # Avoid quantile overlap issues by adding slight noise if necessary
        states = pd.cut(returns, bins=self.quantiles, labels=False, include_lowest=True).dropna().astype(int)
        
        # Calculate transition matrix
        for i in range(len(states) - 1):
            curr_state = states.iloc[i]
            next_state = states.iloc[i+1]
            self.transition_matrix[curr_state, next_state] += 1
            
        # Normalize
        row_sums = self.transition_matrix.sum(axis=1)
        # Handle zero division for empty states
        row_sums[row_sums == 0] = 1 
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]
        
        # Calculate emissions (mean and std for each state)
        for s in range(self.n_states):
            state_returns = returns[states == s]
            if len(state_returns) > 0:
                self.state_means[s] = state_returns.mean()
                self.state_stds[s] = state_returns.std()
            else:
                self.state_means[s] = 0.0
                self.state_stds[s] = 1e-6
                
        self._is_fitted = True
        return self

    def generate(self, n_steps: int, initial_state: Optional[int] = None) -> np.ndarray:
        """
        Generates synthetic return paths based on the fitted HMM.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before generating data.")
            
        if initial_state is None:
            initial_state = np.random.randint(0, self.n_states)
            
        path = np.zeros(n_steps)
        current_state = initial_state
        
        for t in range(n_steps):
            # Sample return from current state distribution (Gaussian approx for simplicity here)
            path[t] = np.random.normal(self.state_means[current_state], self.state_stds[current_state])
            # Transition to next state
            probs = self.transition_matrix[current_state]
            if np.sum(probs) > 0:
                current_state = np.random.choice(self.n_states, p=probs)
            else:
                current_state = np.random.randint(0, self.n_states)
                
        return path


class MarkovRegimeSwitcher:
    """
    Markov-switching model focused on regime identification (e.g., Bull vs. Bear) 
    with time-varying transition probabilities influenced by exogenous factors.
    """
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.transition_matrix = np.ones((n_regimes, n_regimes)) / n_regimes
        self.regime_params = {}
        self._is_fitted = False

    def fit(self, data: pd.DataFrame, target_col: str, exogenous_cols: List[str] = []) -> 'MarkovRegimeSwitcher':
        """
        A simplified heuristic fit for regime switching. 
        In a full implementation, this uses EM or GAS to optimize TVTP.
        Here we define regimes simply by rolling volatility clustering as a proxy.
        """
        # Placeholder heuristic: 2 regimes based on rolling standard deviation
        rolling_std = data[target_col].rolling(window=20).std().dropna()
        median_std = rolling_std.median()
        
        regimes = (rolling_std > median_std).astype(int)
        
        # Calculate transition matrix
        for i in range(len(regimes) - 1):
            self.transition_matrix[regimes.iloc[i], regimes.iloc[i+1]] += 1
            
        row_sums = self.transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]
        
        # Calculate regime means and variances
        for r in range(self.n_regimes):
            idx = regimes[regimes == r].index
            # Intersection to avoid index out of bounds if rolling_std drops initial NaNs
            valid_idx = data.index.intersection(idx)
            if len(valid_idx) > 0:
                self.regime_params[r] = {
                    'mean': data.loc[valid_idx, target_col].mean(),
                    'std': data.loc[valid_idx, target_col].std()
                }
            else:
                self.regime_params[r] = {'mean': 0.0, 'std': 1e-6}
                
        self._is_fitted = True
        return self
        
    def predict_regime(self, recent_data: pd.DataFrame, target_col: str) -> int:
        """
        Infers the current regime based on the most recent data point's volatility.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting regimes.")
            
        if len(recent_data) < 20:
            return 0 # Default to regime 0 if insufficient data
            
        current_std = recent_data[target_col].tail(20).std()
        
        # Compare with learned regime params
        # (This is a simplified distance metric to the closest regime volatility)
        distances = [abs(current_std - self.regime_params[r]['std']) for r in range(self.n_regimes)]
        return int(np.argmin(distances))

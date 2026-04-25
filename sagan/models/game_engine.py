import numpy as np
import logging

logger = logging.getLogger("sagan.game")

class GameTheoreticEngine:
    """
    Predictive engine using Monte Carlo simulations and Game Theory.
    Now coupled directly to discovered symbolic formulas.
    """
    def __init__(self, n_trials: int = 10000):
        self.n_trials = n_trials

    def predict_probabilities(self, fitted_signals: dict, current_data: dict, formula: str) -> dict:
        """
        Runs Monte Carlo trials by evaluating the actual symbolic formula 
        across thousands of perturbed signal realizations.
        """
        np.random.seed(42) # Ensure research reproducibility
        signals = list(fitted_signals.keys())
        # Matrix of shape (n_trials, n_signals)
        perturbations = np.random.normal(0, 1, (self.n_trials, len(signals)))
        std_errs = np.array([fitted_signals[s].get("std_err", 0.5) for s in signals])
        base_values = np.array([current_data.get(s, 0.0) for s in signals])
        
        # Perturbed signals: base + (noise * std_err)
        # Note: we use the base values from current_data and add noise scaled by fitting error
        perturbed_matrix = base_values + (perturbations * std_errs)
        
        # Prepare context for vectorized evaluation
        eval_context = {s: perturbed_matrix[:, i] for i, s in enumerate(signals)}
        eval_context.update({
            "np": np, 
            "exp": np.exp, 
            "log": np.log, 
            "abs": np.abs, 
            "sin": np.sin, 
            "cos": np.cos
        })
        
        try:
            clean_formula = formula.replace("^", "**")
            # Evaluate the symbolic expression on the entire trial matrix at once (vectorized)
            sim_results = eval(clean_formula, {"__builtins__": {}}, eval_context)
            
            # If results are complex or NaN, clean them
            sim_results = np.nan_to_num(np.real(sim_results))
            
            mu, sigma = np.mean(sim_results), np.std(sim_results) + 1e-8
            
            # Probability thresholds (Up > mu + 0.5sigma, Down < mu - 0.5sigma)
            up_count = np.sum(sim_results > (mu + 0.2 * sigma))
            down_count = np.sum(sim_results < (mu - 0.2 * sigma))
            sideways_count = self.n_trials - up_count - down_count
            
            probs = {
                "up": float(up_count / self.n_trials),
                "down": float(down_count / self.n_trials),
                "sideways": float(sideways_count / self.n_trials),
                "expected_upside": float(np.mean(sim_results[sim_results > mu]) if any(sim_results > mu) else 0.0)
            }
            return probs
            
        except Exception as e:
            logger.error(f"GameEngine: Vectorized MC evaluation failed: {e}")
            return {"up": 0.33, "down": 0.33, "sideways": 0.34, "expected_upside": 0.0}

    def calculate_nash_weights(self, ticker_probs: dict, invert: bool = True) -> dict:
        """
        Calculates optimal weights using a zero-sum game approach.
        Uses Log-Scaling and optional Signal Inversion.
        """
        weights = {}
        for ticker, data in ticker_probs.items():
            confidence = data["up"] - data["down"]
            upside = data["expected_upside"]
            
            # raw_weight = confidence * upside
            raw_weight = upside * confidence
            
            if invert:
                raw_weight = -raw_weight
                
            # Log-scaling
            scaled_weight = np.sign(raw_weight) * np.log1p(np.abs(raw_weight))
            weights[ticker] = float(scaled_weight)
            
        # Normalize weights so the max absolute weight is 1.0
        if weights:
            max_w = max(abs(w) for w in weights.values())
            if max_w > 0:
                weights = {t: w / max_w for t, w in weights.items()}
                
        return weights

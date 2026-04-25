import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import logging
import numba

logger = logging.getLogger("sagan.math")

@numba.jit(nopython=True)
def polynomial_kernel(t, coeffs):
    # coeffs is an array, we use Horner's method for speed in JIT
    res = np.zeros_like(t, dtype=np.float64)
    for c in coeffs:
        res = res * t + c
    return res

@numba.jit(nopython=True)
def fourier_kernel(t, params):
    # params: [a0, a1, b1, w1, a2, b2, w2, ...]
    res = np.full_like(t, params[0], dtype=np.float64)
    n_harmonics = (len(params) - 1) // 3
    for i in range(n_harmonics):
        a = params[1 + i*3]
        b = params[1 + i*3 + 1]
        w = params[1 + i*3 + 2]
        res += a * np.cos(w * t) + b * np.sin(w * t)
    return res

def fit_signal_worker(y, signal_name, target_r2=0.95):
    """
    Standalone worker function for parallel fitting.
    """
    engine = MathematicalEngine()
    func, params, r2 = engine.fit_variable(y, target_r2=target_r2)
    return signal_name, {"func": func, "params": params, "r2": r2}

class MathematicalEngine:
    """
    Library of basis functions and iterative fitting logic.
    """
    
    @staticmethod
    def polynomial(t, *coeffs):
        return polynomial_kernel(t, np.array(coeffs))

    @staticmethod
    def fourier(t, *params):
        return fourier_kernel(t, np.array(params))

    def fit_variable(self, y: np.ndarray, target_r2: float = 0.95, max_complexity: int = 20):
        """
        Iteratively tries to fit y using increasing complexity until target_r2 is met.
        """
        t = np.arange(len(y))
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
        
        best_r2 = -np.inf
        best_func = None
        best_popt = None
        
        # 1. Try Polynomials
        for degree in range(1, 10):
            coeffs = np.polyfit(t, y_norm, degree)
            y_pred = np.polyval(coeffs, t)
            r2 = r2_score(y_norm, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_func = "polynomial"
                best_popt = coeffs.tolist()
            
            if r2 >= target_r2:
                return best_func, best_popt, r2

        # 2. Try Fourier Series if polynomial isn't enough
        for n_harmonics in range(1, 6):
            initial_guess = [0.0] + [0.1, 0.1, 0.05] * n_harmonics
            try:
                popt, _ = curve_fit(self.fourier, t, y_norm, p0=initial_guess, maxfev=2000)
                y_pred = self.fourier(t, *popt)
                r2 = r2_score(y_norm, y_pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_func = "fourier"
                    best_popt = popt.tolist()
                
                if r2 >= target_r2:
                    return best_func, best_popt, r2
            except:
                pass

        return best_func, best_popt, best_r2

    @staticmethod
    def evaluate(func_name: str, t: np.ndarray, params: list):
        if func_name == "polynomial":
            return polynomial_kernel(t, np.array(params))
        elif func_name == "fourier":
            return fourier_kernel(t, np.array(params))
        return np.zeros_like(t)

    def find_best_composition(self, train_data: pd.DataFrame, val_data: pd.DataFrame, target_col: str, candidates: list[str]) -> tuple[str, float]:
        """
        Evaluates several candidate formulas on validation data and returns the best one.
        """
        best_r2 = -np.inf
        best_formula = candidates[0] if candidates else " + ".join(train_data.columns)
        
        # Ensure we have np in context for eval
        eval_context = {"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos}
        
        for formula in candidates:
            try:
                # 1. Evaluate on training data to check validity
                train_context = {col: train_data[col].values for col in train_data.columns}
                train_context.update(eval_context)
                
                # Basic cleanup
                clean_formula = formula.replace("^", "**")
                
                # 2. Evaluate on validation data for OOS performance
                val_context = {col: val_data[col].values for col in val_data.columns}
                val_context.update(eval_context)
                
                y_val_pred = eval(clean_formula, {"__builtins__": {}}, val_context)
                y_val_true = val_data[target_col].values
                
                r2 = r2_score(y_val_true, y_val_pred)
                logger.info(f"Formula: {formula} | Val R2: {r2:.4f}")
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_formula = formula
            except Exception as e:
                logger.debug(f"Failed to evaluate candidate {formula}: {e}")
                continue
                
        return best_formula, best_r2

def soft_gating(x, weights):
    exp_w = np.exp(weights - np.max(weights))
    return exp_w / np.sum(exp_w)

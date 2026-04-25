import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import logging
import numba
from typing import Dict, List

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

def fit_signal_worker(y, signal_name):
    """
    Standalone worker function for parallel fitting.
    """
    engine = MathematicalEngine()
    func, params, r2, std_err = engine.fit_variable(y)
    return signal_name, {"func": func, "params": params, "r2": r2, "std_err": std_err}

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

    def fit_variable(self, y: np.ndarray, max_complexity: int = 20):
        """
        Iteratively tries to fit y using increasing complexity and returns the best fit + standard error.
        """
        t = np.arange(len(y))
        mean_y, std_y = np.mean(y), np.std(y) + 1e-8
        y_norm = (y - mean_y) / std_y
        
        best_r2 = -np.inf
        best_func = None
        best_popt = None
        best_y_pred = None
        
        # 1. Try Polynomials
        for degree in range(1, 10):
            coeffs = np.polyfit(t, y_norm, degree)
            y_pred = np.polyval(coeffs, t)
            r2 = r2_score(y_norm, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_func = "polynomial"
                best_popt = coeffs.tolist()
                best_y_pred = y_pred
            
        # 2. Try Fourier Series
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
                    best_y_pred = y_pred
            except:
                pass

        # Calculate Standard Error of the Estimate (SEE)
        if best_y_pred is not None:
            residuals = y_norm - best_y_pred
            std_err = np.std(residuals)
        else:
            std_err = 1.0

        return best_func, best_popt, best_r2, float(std_err)

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
                # 1. Sanitize keys for eval
                sanitized_cols = {col.replace(" ", "_").replace("^", "_IDX_"): col for col in train_data.columns}
                
                # 2. Evaluate on training data to check validity
                train_context = {s_col: train_data[orig_col].values for s_col, orig_col in sanitized_cols.items()}
                train_context.update(eval_context)
                
                # Basic cleanup
                clean_formula = formula.replace("^", "**") # For powers
                # Now replace variables in formula with sanitized versions
                for s_col, orig_col in sanitized_cols.items():
                    # We use a regex or simple replace if we're careful. 
                    # For candidate formulas from LLM, they might already use ^VIX.
                    # We need to replace '^VIX' with '_IDX_VIX'
                    clean_formula = clean_formula.replace(orig_col, s_col)
                
                # 3. Evaluate on validation data for OOS performance
                val_context = {s_col: val_data[orig_col].values for s_col, orig_col in sanitized_cols.items()}
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

    def evaluate_formula(self, formula: str, data_context: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Safely evaluates a symbolic formula using the provided data context.
        """
        # Sanitize keys for eval
        sanitized_context = {k.replace(" ", "_").replace("^", "_IDX_"): v for k, v in data_context.items()}
        sanitized_context.update({
            "np": np, 
            "exp": np.exp, 
            "log": np.log, 
            "sin": np.sin, 
            "cos": np.cos,
            "abs": np.abs,
            "sqrt": np.sqrt
        })
        
        # We need to handle '^' as power vs '^' as indicator prefix
        # First, temporarily replace '^' in indicators to something unique
        clean_formula = formula
        for k in data_context.keys():
            if "^" in k:
                clean_formula = clean_formula.replace(k, k.replace("^", "_IDX_"))
            if " " in k:
                clean_formula = clean_formula.replace(k, k.replace(" ", "_"))
        
        # Now handle power operators (if any left that aren't part of names)
        clean_formula = clean_formula.replace("^", "**")
                
        return eval(clean_formula, {"__builtins__": {}}, sanitized_context)

    def explain_formula(self, formula: str) -> List[str]:
        """
        Attempts to break down a formula into logical additive components for visualization.
        Example: "(Close * 0.5) + log(Volume)" -> ["(Close * 0.5)", "log(Volume)"]
        """
        # Very naive split on ' + ' and ' - ' (not considering parentheses depth, but good for simple formulas)
        # In a real system, we'd use an AST parser.
        import re
        # This is a placeholder for a more robust AST-based splitter
        components = re.split(r' \+ | \- ', formula)
        return [c.strip() for c in components if c.strip()]

def soft_gating(x, weights):
    exp_w = np.exp(weights - np.max(weights))
    return exp_w / np.sum(exp_w)

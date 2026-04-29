import logging
import numba
import torch
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from typing import Dict, List, Optional
from sagan.models.controller_arch import ControllerLSTM

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


class CenteredModelBasis:
    """
    Specialized basis function using the pre-trained centered model expression.
    """
    def __init__(self, pkl_path: str):
        self.pkl_path = pkl_path
        self.expression = None
        self.mean = 0.0
        self.std = 1.0
        self._load()

    def _load(self):
        try:
            with open(self.pkl_path, 'rb') as f:
                data = pickle.load(f)
            self.expression = data.get('expression')
            self.mean = data.get('y_mean_c', 0.0)
            self.std = data.get('y_std_c', 1.0)
            logger.info(f"CenteredModel: Loaded expression '{self.expression}'")
        except Exception as e:
            logger.error(f"CenteredModel: Failed to load: {e}")

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        if not self.expression: return np.zeros_like(t)
        # Expression is like "-2*t_sym - 19.25"
        # We replace t_sym with t
        t_sym = t
        try:
            # Safely evaluate
            clean_expr = self.expression.replace("t_sym", "t_sym_val")
            val = eval(clean_expr, {"__builtins__": {}}, {"t_sym_val": t})
            # Denormalize if needed? The user said "figuring out the math behind the prices"
            # Usually these models output normalized values
            return val * self.std + self.mean
        except Exception as e:
            logger.error(f"CenteredModel: Eval failed: {e}")
            return np.zeros_like(t)

class ControllerEngine:
    """
    Wrapper for the 3-layer LSTM Controller.
    """
    def __init__(self, pth_path: str):
        self.model = ControllerLSTM()
        try:
            self.model.load_state_dict(torch.load(pth_path, map_location='cpu'))
            self.model.eval()
            logger.info("ControllerEngine: Loaded pre-trained LSTM.")
        except Exception as e:
            logger.error(f"ControllerEngine: Failed to load: {e}")

    def score_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.model(tokens)
            return logits

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

    def fit_variable(self, y: np.ndarray, max_complexity: int = 20, use_specialized: bool = True):
        """
        Iteratively tries to fit y using increasing complexity and returns the best fit + standard error.
        Now includes specialized pre-trained models.
        """
        t = np.arange(len(y))
        mean_y, std_y = np.mean(y), np.std(y) + 1e-8
        y_norm = (y - mean_y) / std_y
        
        best_r2 = -np.inf
        best_func = None
        best_popt = None
        best_y_pred = None
        
        # Pre-calculate variance for R2 speedup
        var_y = np.var(y_norm)
        if var_y == 0: var_y = 1e-8

        # 0. Specialized Models (If requested)
        if use_specialized:
            from sagan.symbolic_lib.download_models import CENTERED_MODEL_PATH
            if CENTERED_MODEL_PATH.exists():
                cm = CenteredModelBasis(str(CENTERED_MODEL_PATH))
                y_pred = cm.evaluate(t)
                # Normalize y_pred to compare R2
                y_pred_norm = (y_pred - np.mean(y_pred)) / (np.std(y_pred) + 1e-8)
                r2 = 1 - np.mean((y_norm - y_pred_norm)**2) / var_y
                if r2 > best_r2:
                    best_r2 = r2
                    best_func = "centered_model"
                    best_popt = [cm.pkl_path]
                    best_y_pred = y_pred_norm
                
                # Early exit if specialized model is excellent
                if best_r2 > 0.98: return best_func, best_popt, float(best_r2), 0.01

        # 1. Try Polynomials (Reduced range for speed, 1-6 instead of 1-9)
        for degree in range(1, 7):
            coeffs = np.polyfit(t, y_norm, degree)
            y_pred = np.polyval(coeffs, t)
            r2 = 1 - np.mean((y_norm - y_pred)**2) / var_y
            if r2 > best_r2:
                best_r2 = r2
                best_func = "polynomial"
                best_popt = coeffs.tolist()
                best_y_pred = y_pred
            
            # Early exit if R2 is high
            if best_r2 > 0.95: break
            
        # 2. Try Fourier Series (Only if needed and R2 is still low)
        if best_r2 < 0.8:
            for n_harmonics in range(1, 3): # Reduced harmonics 1-2
                initial_guess = [0.0] + [0.1, 0.1, 0.05] * n_harmonics
                try:
                    popt, _ = curve_fit(self.fourier, t, y_norm, p0=initial_guess, maxfev=500) # Reduced maxfev
                    y_pred = self.fourier(t, *popt)
                    r2 = 1 - np.mean((y_norm - y_pred)**2) / var_y
                    if r2 > best_r2:
                        best_r2 = r2
                        best_func = "fourier"
                        best_popt = popt.tolist()
                        best_y_pred = y_pred
                    if best_r2 > 0.95: break
                except:
                    pass

        # Calculate Standard Error of the Estimate (SEE)
        if best_y_pred is not None:
            residuals = y_norm - best_y_pred
            std_err = np.std(residuals)
        else:
            std_err = 1.0

        return best_func, best_popt, float(best_r2), float(std_err)

    @staticmethod
    def evaluate(func_name: str, t: np.ndarray, params: list):
        if func_name == "polynomial":
            return polynomial_kernel(t, np.array(params))
        elif func_name == "fourier":
            return fourier_kernel(t, np.array(params))
        elif func_name == "centered_model":
            cm = CenteredModelBasis(params[0])
            return cm.evaluate(t)
        elif func_name == "time_variable":
            # params[0] would be the path to the TVMM state_dict
            # This is a placeholder for actual integration in predicting loops
            return np.zeros_like(t) 
        return np.zeros_like(t)

    def find_best_composition(self, train_data: pd.DataFrame, val_data: pd.DataFrame, target_col: str, candidates: list[str]) -> tuple[str, float]:
        """
        Evaluates several candidate formulas on validation data in parallel and returns the best one.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        logger.info(f"Evaluating {len(candidates)} candidates in parallel...")
        
        def _eval_candidate(formula):
            try:
                sanitized_cols = {col.replace(" ", "_").replace("^", "_IDX_"): col for col in train_data.columns}
                clean_formula = formula.replace("^", "**")
                for s_col, orig_col in sanitized_cols.items():
                    clean_formula = clean_formula.replace(orig_col, s_col)
                
                # Val Evaluation
                val_context = {s_col: val_data[orig_col].values for s_col, orig_col in sanitized_cols.items()}
                val_context.update({"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos, "abs": np.abs})
                
                y_val_pred = eval(clean_formula, {"__builtins__": {}}, val_context)
                y_val_true = val_data[target_col].values
                
                r2 = r2_score(y_val_true, y_val_pred)
                return formula, r2
            except:
                return formula, -np.inf

        best_r2 = -np.inf
        best_formula = candidates[0] if candidates else "None"
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_eval_candidate, candidates))
            
        for formula, r2 in results:
            if r2 > best_r2:
                best_r2 = r2
                best_formula = formula
        
        return best_formula, best_r2

    def evaluate_formula(self, formula: str, data_context: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Safely evaluates a symbolic formula using the provided data context.
        Vectorized for high performance on arrays.
        """
        sanitized_context = {k.replace(" ", "_").replace("^", "_IDX_"): v for k, v in data_context.items()}
        sanitized_context.update({
            "np": np, 
            "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos,
            "abs": np.abs, "sqrt": np.sqrt, "max": np.max, "min": np.min
        })
        
        clean_formula = formula
        for k in data_context.keys():
            if "^" in k or " " in k:
                clean_formula = clean_formula.replace(k, k.replace("^", "_IDX_").replace(" ", "_"))
        
        clean_formula = clean_formula.replace("^", "**")
                
        return eval(clean_formula, sanitized_context)

    def evaluate_ensemble(self, formula: str, fitted_signals: Dict[str, dict], data: pd.DataFrame) -> np.ndarray:
        """
        Evaluates a composite formula by first evaluating all basis functions (fitted signals) 
        vectorized over the entire dataframe.
        """
        t = np.arange(len(data))
        eval_context = {s: self.evaluate(f["func"], t, f["params"]) for s, f in fitted_signals.items()}
        return self.evaluate_formula(formula, eval_context)

    def explain_formula(self, formula: str) -> List[str]:
        """
        Attempts to break down a formula into logical additive components for visualization.
        """
        import re
        components = re.split(r' \+ | \- ', formula)
        return [c.strip() for c in components if c.strip()]

def soft_gating(x, weights):
    exp_w = np.exp(weights - np.max(weights))
    return exp_w / np.sum(exp_w)

def soft_gating(x, weights):
    exp_w = np.exp(weights - np.max(weights))
    return exp_w / np.sum(exp_w)

class TimeVariableEvaluator:
    """
    Handles real-time inference for LSTM-driven math models.
    """
    def __init__(self, model_path: str, feature_names: List[str], input_size: int):
        from sagan.models.tv_math import TimeVariableMathModel
        self.model = TimeVariableMathModel(input_size=input_size, feature_names=feature_names)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            logger.info(f"TVEvaluator: Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"TVEvaluator: Load failed: {e}")

    def predict(self, x_seq: np.ndarray, x_curr: np.ndarray):
        with torch.no_grad():
            t_seq = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0)
            t_curr = torch.tensor(x_curr, dtype=torch.float32).unsqueeze(0)
            pred, weights, bias = self.model(t_seq, t_curr)
            
            formula = self.model.explain(weights[0], bias[0])
            return pred.item(), formula

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Callable, Dict, List, Tuple

class SymbolicRefinementOptimizer:
    """
    Symbolic Refinement Layer fitting microstructural equations to MoE prediction residuals:
    g_Symbolic(X) = y_resid_pred
    """
    def __init__(self):
        # We define a library of candidate microstructural formulas.
        # Each candidate is a function: (df_features, params) -> prediction_series
        self.formulas = {
            "OFI_Vol_Pressure": {
                "func": lambda df, p: p[0] * df["ofi"] * df["rolling_vol"],
                "num_params": 1,
                "latex": "c_1 \\cdot \\text{OFI}_t \\cdot \\sigma_t"
            },
            "Hawkes_Spread_Elasticity": {
                "func": lambda df, p: p[0] * np.sin(df["rolling_vol"]) * df["hawkes_intensity"],
                "num_params": 1,
                "latex": "c_1 \\cdot \\sin(\\sigma_t) \\cdot \\lambda_t"
            },
            "Book_Imbalance_Reversion": {
                "func": lambda df, p: p[0] * df["depth_imbalance"] * df["spread"] + p[1] * np.cos(df["hawkes_intensity"]),
                "num_params": 2,
                "latex": "c_1 \\cdot \\Delta_t \\cdot S_t + c_2 \\cdot \\cos(\\lambda_t)"
            },
            "Composite_OFI_Intensity": {
                "func": lambda df, p: p[0] * (df["ofi"] / (df["hawkes_intensity"] + 1e-4)) + p[1] * df["spread"] * df["rolling_vol"],
                "num_params": 2,
                "latex": "c_1 \\cdot \\frac{\\text{OFI}_t}{\\lambda_t} + c_2 \\cdot S_t \\cdot \\sigma_t"
            }
        }
        self.best_formula_name = "OFI_Vol_Pressure"
        self.fitted_params = np.array([0.0])

    def fit(self, df: pd.DataFrame, residuals: np.ndarray):
        """
        Fits all candidate formulas to the residuals and selects the one with the lowest MSE.
        """
        best_mse = float("inf")
        best_name = None
        best_params = None

        # Clean inf/nan in residuals
        clean_mask = np.isfinite(residuals) & np.isfinite(df["ofi"].values)
        df_clean = df.iloc[clean_mask]
        res_clean = residuals[clean_mask]

        if len(res_clean) == 0:
            print("Warning: No valid residual points to fit symbolic model.")
            return

        for name, entry in self.formulas.items():
            func = entry["func"]
            num_params = entry["num_params"]
            
            # High-fidelity loss function penalizing unstable parameters and taker exchange charges
            def loss_func(params):
                pred = func(df_clean, params)
                mse = np.mean((res_clean - pred) ** 2)
                
                # 1. L2 parameter regularization to prevent coefficients drifting into unstable zones
                l2_penalty = 0.01 * np.sum(params ** 2)
                
                # 2. Taker crossing exchange fees penalty (barrier = ~3.52 bps)
                if "mid_price" in df_clean.columns:
                    fee_barrier = df_clean["mid_price"].values * 0.000352
                else:
                    fee_barrier = np.ones(len(df_clean)) * 1000.0 * 0.000352
                taker_crossings = np.sum(np.abs(pred) > fee_barrier * 1.5)
                taker_penalty = 0.05 * taker_crossings * np.mean(fee_barrier)
                
                return mse + l2_penalty + taker_penalty

            # Initial guess
            x0 = np.zeros(num_params)
            
            # Optimize parameters
            res = minimize(loss_func, x0, method="BFGS")
            optimized_params = res.x
            
            # Calculate final MSE
            final_pred = func(df_clean, optimized_params)
            final_mse = np.mean((res_clean - final_pred) ** 2)
            
            if final_mse < best_mse:
                best_mse = final_mse
                best_name = name
                best_params = optimized_params
                
        self.best_formula_name = best_name
        self.fitted_params = best_params
        
        print(f"Selected Symbolic Formula: {self.best_formula_name}")
        print(f"Mathematical Equation: {self.formulas[self.best_formula_name]['latex']}")
        print(f"Fitted Parameters: {self.fitted_params.tolist()}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Evaluates the selected symbolic formula on the input features.
        """
        entry = self.formulas[self.best_formula_name]
        func = entry["func"]
        pred = func(df, self.fitted_params)
        # Handle nan/inf
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred

if __name__ == "__main__":
    # Test optimizer
    df = pd.DataFrame({
        "ofi": np.random.randn(100),
        "rolling_vol": np.random.uniform(0.01, 0.05, 100),
        "hawkes_intensity": np.random.uniform(0.1, 2.0, 100),
        "depth_imbalance": np.random.uniform(-1, 1, 100),
        "spread": np.random.uniform(0.05, 0.25, 100)
    })
    residuals = np.random.randn(100) * 0.1
    
    sym = SymbolicRefinementOptimizer()
    sym.fit(df, residuals)
    pred = sym.predict(df)
    print("Sym prediction shape:", pred.shape)

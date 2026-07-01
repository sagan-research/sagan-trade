import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

class SymbolicRegressor:
    """
    Symbolic Regressor discovering mathematical representations of alpha signals.
    Supports standard indicators, limit order book metrics, and custom mathematical functions.
    """
    def __init__(self, basis_functions=None):
        self.basis_functions = basis_functions or ['poly', 'fourier']
        self.best_formula_name = None
        self.fitted_params = None
        self.data = None
        self.signals = []
        self.target = None

        # Library of mathematical candidates for fitting
        self.formulas = {
            "Poly_Signal": {
                "func": lambda df, p: p[0] * df["Close"] + p[1] * df["RSI"] * 0.01,
                "num_params": 2,
                "latex": "c_1 \\cdot P_t + c_2 \\cdot \\text{RSI}_t"
            },
            "Fourier_Signal": {
                "func": lambda df, p: p[0] * np.sin(df["Close"] / (df["Close"].rolling(20).mean() + 1e-8)) + p[1] * np.cos(df["RSI"] * 0.05),
                "num_params": 2,
                "latex": "c_1 \\cdot \\sin(\\tilde{P}_t) + c_2 \\cdot \\cos(0.05 \\cdot \\text{RSI}_t)"
            },
            "Momentum_Volume_Signal": {
                "func": lambda df, p: p[0] * (df["Close"] - df["Close"].shift(10).bfill()) * np.log(df["Volume"] + 1e-4),
                "num_params": 1,
                "latex": "c_1 \\cdot \\Delta_{10} P_t \\cdot \\ln(V_t)"
            },
            "LOB_OFI_Vol_Pressure": {
                "func": lambda df, p: p[0] * df.get("ofi", pd.Series(0.0, index=df.index)) * df.get("rolling_vol", pd.Series(0.0, index=df.index)),
                "num_params": 1,
                "latex": "c_1 \\cdot \\text{OFI}_t \\cdot \\sigma_t"
            }
        }

    def _compute_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.clip(lower=0)).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def train(self, target, signals, data=None):
        """
        Fits candidate symbolic formulas to target return dynamics.
        target: ticker symbol string or target pandas Series
        signals: list of signal names
        data: optional pandas DataFrame containing target and signals
        """
        self.target = target
        self.signals = signals

        if data is None:
            if isinstance(target, str):
                print(f"Downloading historical data for target '{target}' via yfinance...")
                df = yf.download(target, period="2y", progress=False)
                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                self.data = df
            else:
                raise ValueError("If 'data' is not provided, 'target' must be a ticker symbol string.")
        else:
            self.data = data.copy()

        # Compute technical indicators if requested but missing
        if "RSI" in signals and "RSI" not in self.data.columns:
            if "Close" in self.data.columns:
                self.data["RSI"] = self._compute_rsi(self.data["Close"]).fillna(50)
            elif isinstance(target, pd.Series):
                self.data["RSI"] = self._compute_rsi(target).fillna(50)
            else:
                self.data["RSI"] = 50.0

        # Define targets: next-day returns
        if "Close" in self.data.columns:
            y = self.data["Close"].pct_change().shift(-1).fillna(0.0).values
        else:
            y = self.data.iloc[:, 0].pct_change().shift(-1).fillna(0.0).values

        # Select compatible formulas based on available columns
        compatible_formulas = {}
        for name, item in self.formulas.items():
            # Check if all required columns in lambda are in self.data
            required_cols = []
            if name == "Poly_Signal" or name == "Fourier_Signal":
                required_cols = ["Close", "RSI"]
            elif name == "Momentum_Volume_Signal":
                required_cols = ["Close", "Volume"]
            elif name == "LOB_OFI_Vol_Pressure":
                required_cols = ["ofi", "rolling_vol"]
            
            if all(col in self.data.columns for col in required_cols):
                compatible_formulas[name] = item

        if not compatible_formulas:
            # Fallback to general linear combo of available signals
            self.data["Close"] = self.data.get("Close", self.data.iloc[:, 0])
            self.data["RSI"] = self.data.get("RSI", pd.Series(50.0, index=self.data.index))
            compatible_formulas["Poly_Signal"] = self.formulas["Poly_Signal"]

        best_mse = float("inf")
        best_name = None
        best_params = None

        for name, entry in compatible_formulas.items():
            func = entry["func"]
            num_params = entry["num_params"]
            
            def loss_func(params):
                pred = func(self.data, params)
                # MSE of prediction vs. next-day return
                return np.mean((y - pred) ** 2)

            x0 = np.zeros(num_params)
            res = minimize(loss_func, x0, method="Nelder-Mead")
            
            final_pred = func(self.data, res.x)
            final_mse = np.mean((y - final_pred) ** 2)
            
            if final_mse < best_mse:
                best_mse = final_mse
                best_name = name
                best_params = res.x

        self.best_formula_name = best_name
        self.fitted_params = best_params
        
        print(f"Discovered Symbolic Strategy: {self.best_formula_name}")
        print(f"Formula Equation (LaTeX): {self.formulas[self.best_formula_name]['latex']}")
        print(f"Fitted Parameters: {self.fitted_params.tolist()}")
        return f"model_symbolic_{best_name}"

    def predict(self, data=None):
        """
        Generates predicted alpha signals and returns the formula representation.
        """
        df = self.data if data is None else data.copy()
        if df is None:
            raise ValueError("No data available for prediction. Run 'train' first or pass 'data'.")

        if "RSI" in self.signals and "RSI" not in df.columns:
            if "Close" in df.columns:
                df["RSI"] = self._compute_rsi(df["Close"]).fillna(50)
            else:
                df["RSI"] = 50.0

        entry = self.formulas[self.best_formula_name]
        func = entry["func"]
        pred = func(df, self.fitted_params)
        
        # Handle nan/inf
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Format the formula string with parameter values
        formula_latex = entry["latex"]
        formula_str = formula_latex
        for i, val in enumerate(self.fitted_params):
            formula_str = formula_str.replace(f"c_{i+1}", f"{val:.6f}")
            
        return pd.Series(pred, index=df.index), formula_str

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Lasso
from sagan.models.symbolic_fitter import LSTMSymbolicFitter

class LSTMRobustFitter(LSTMSymbolicFitter):
    """
    Robust 5-layer LSTM Fitter with L1-LASSO Sparsification.
    Ensures the math "sticks" by enforcing Occam's Razor.
    """
    def __init__(self, input_size=1, hidden_size=256, n_harmonics=10, alpha=0.01):
        # We increase n_harmonics to provide a richer dictionary, 
        # then LASSO will prune it.
        super().__init__(input_size, hidden_size, n_harmonics)
        self.alpha = alpha # LASSO regularization strength
        self.n_harmonics = n_harmonics

    def fit_sparse(self, t, y_true):
        """
        Fits the basis functions using LASSO to ensure sparsity.
        t: Time grid (normalized 0-1)
        y_true: Stationary target signal
        """
        # 1. Get basis functions
        # For simplicity, we'll build a matrix of basis function values
        X_basis = []
        
        # Poly terms
        X_basis.append(np.ones_like(t))
        X_basis.append(t)
        X_basis.append(t**2)
        
        # Fourier terms
        freqs = np.linspace(0.1, 10.0, self.n_harmonics)
        for w in freqs:
            X_basis.append(np.cos(w * t))
            X_basis.append(np.sin(w * t))
            
        # Non-linear terms (for non-periodic patterns)
        # We add small epsilon to t for log/sqrt to avoid domain errors
        t_eps = t + 1e-6
        X_basis.append(np.exp(t))
        X_basis.append(np.log(t_eps))
        X_basis.append(np.sqrt(t_eps))
        X_basis.append(np.abs(t - 0.5))
        
        X_basis = np.array(X_basis).T # (T, Num_Basis)
        
        # 2. Run LASSO
        lasso = Lasso(alpha=self.alpha, max_iter=5000)
        lasso.fit(X_basis, y_true)
        
        return lasso.coef_, lasso.intercept_, X_basis, freqs

    def get_sparse_formula(self, coefs, intercept, freqs):
        """
        Returns a simplified formula containing only non-zero terms.
        """
        terms = []
        if abs(intercept) > 1e-4:
            terms.append(f"{intercept:.4f}")
            
        # Poly
        if abs(coefs[1]) > 1e-4: terms.append(f"({coefs[1]:.4f} * t)")
        if abs(coefs[2]) > 1e-4: terms.append(f"({coefs[2]:.4f} * t^2)")
        
        # Fourier
        idx = 3
        for w in freqs:
            A = coefs[idx]
            B = coefs[idx+1]
            if abs(A) > 1e-4: terms.append(f"({A:.4f} * cos({w:.4f}*t))")
            if abs(B) > 1e-4: terms.append(f"({B:.4f} * sin({w:.4f}*t))")
            idx += 2
            
        # Non-linear
        if abs(coefs[idx]) > 1e-4: terms.append(f"({coefs[idx]:.4f} * exp(t))")
        if abs(coefs[idx+1]) > 1e-4: terms.append(f"({coefs[idx+1]:.4f} * log(t))")
        if abs(coefs[idx+2]) > 1e-4: terms.append(f"({coefs[idx+2]:.4f} * sqrt(t))")
        if abs(coefs[idx+3]) > 1e-4: terms.append(f"({coefs[idx+3]:.4f} * abs(t-0.5))")
            
        return " + ".join(terms) if terms else "0.0"

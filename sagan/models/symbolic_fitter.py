import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sagan.models.controller_arch import ControllerLSTM

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        # x is (B, C, T)
        x = self.conv(x)
        # remove trailing padding to make it causal
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class BaseSymbolicFitter(nn.Module):
    @staticmethod
    def evaluate_math(t, coeffs, n_harmonics=3):
        """
        Evaluate the math function at time t given coefficients.
        coeffs: [a0, a1, a2, A1, B1, w1, A2, B2, w2, ...]
        """
        # Poly part
        a0, a1, a2 = coeffs[0], coeffs[1], coeffs[2]
        res = a0 + a1 * t + a2 * (t**2)
        
        # Fourier part
        for i in range(n_harmonics):
            A = coeffs[3 + i*3]
            B = coeffs[3 + i*3 + 1]
            w = coeffs[3 + i*3 + 2]
            res += A * torch.cos(w * t) + B * torch.sin(w * t)
            
        return res

    def get_formula(self, coeffs, n_harmonics=3):
        """
        Returns a human-readable string of the fitted math function.
        """
        c = coeffs.detach().cpu().numpy()
        terms = [f"{c[0]:.4f}", f"({c[1]:.4f} * t)", f"({c[2]:.4f} * t^2)"]
        
        for i in range(n_harmonics):
            A, B, w = c[3+i*3], c[3+i*3+1], c[3+i*3+2]
            terms.append(f"({A:.4f} * cos({w:.4f}*t))")
            terms.append(f"({B:.4f} * sin({w:.4f}*t))")
            
        return " + ".join(terms)

class LegacyLSTMSymbolicFitter(BaseSymbolicFitter):
    """
    5-layer LSTM Fitter that maps price sequences to mathematical coefficients.
    Legacy compute-heavy architecture.
    """
    def __init__(self, input_size=1, hidden_size=256, n_harmonics=3):
        super().__init__()
        self.num_coeffs = 3 + 3 * n_harmonics
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, self.num_coeffs)
        
    def forward(self, x):
        # x: (B, T, 1)
        _, (h, _) = self.lstm(x)
        top_h = h[-1] # (B, H)
        coeffs = self.fc(top_h) # (B, Num_Coeffs)
        return coeffs

class HybridSymbolicFitter(BaseSymbolicFitter):
    """
    Compute-light architecture utilizing 1D Convolutions and 2-layer GRU.
    Designed for high throughput and sequence parallelization.
    """
    def __init__(self, input_size=1, hidden_size=256, n_harmonics=3):
        super().__init__()
        self.num_coeffs = 3 + 3 * n_harmonics
        
        # Lightweight feature extraction across time
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_size // 2, out_channels=hidden_size, kernel_size=3, padding=1)
        
        # 2-layer GRU instead of 5-layer LSTM
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, self.num_coeffs)
        
    def forward(self, x):
        # x: (B, T, C)
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2) # Back to (B, T, C)
        
        _, h = self.gru(x) # h: (num_layers, B, H)
        top_h = h[-1] # (B, H)
        coeffs = self.fc(top_h)
        return coeffs

class TCNSymbolicFitter(BaseSymbolicFitter):
    """
    Temporal Convolutional Network Fitter.
    Highly parallelized causal convolutions for extreme compute efficiency.
    """
    def __init__(self, input_size=1, hidden_size=128, n_harmonics=3):
        super().__init__()
        self.num_coeffs = 3 + 3 * n_harmonics
        
        # Exponentially increasing receptive field: dilation = 1, 2, 4, 8
        self.tcn = nn.Sequential(
            CausalConv1d(input_size, hidden_size, kernel_size=3, dilation=1),
            nn.ReLU(),
            CausalConv1d(hidden_size, hidden_size, kernel_size=3, dilation=2),
            nn.ReLU(),
            CausalConv1d(hidden_size, hidden_size, kernel_size=3, dilation=4),
            nn.ReLU(),
            CausalConv1d(hidden_size, hidden_size, kernel_size=3, dilation=8),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_size, self.num_coeffs)
        
    def forward(self, x):
        # x: (B, T, C) -> TCN expects (B, C, T)
        x = x.transpose(1, 2)
        out = self.tcn(x) # (B, hidden_size, T)
        
        # Take the feature representation of the LAST time step
        top_h = out[:, :, -1] # (B, hidden_size)
        coeffs = self.fc(top_h)
        return coeffs

# Default alias for the active architecture
LSTMSymbolicFitter = TCNSymbolicFitter

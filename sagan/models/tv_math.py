import torch
import torch.nn as nn
import numpy as np
from sagan.models.lstm_direct import DirectLSTM

class TimeVariableMathModel(nn.Module):
    """
    Time-Variable Math Model (TVMM).
    Uses a 5-layer LSTM to predict parameters of a symbolic expression.
    USP: Explainability via time-varying coefficients.
    """
    def __init__(self, input_size, feature_names, hidden_size=256):
        super().__init__()
        self.feature_names = feature_names
        self.num_params = len(feature_names) + 1  # Weights for each feature + bias
        
        self.lstm_brain = DirectLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=self.num_params,
            num_layers=5
        )
        
    def forward(self, x_seq, x_current):
        """
        x_seq: (B, Window, Input_Size) - Historical context for LSTM
        x_current: (B, Num_Features) - Current features to apply the math model to
        """
        # 1. Brain predicts the parameters for the current state
        params = self.lstm_brain(x_seq) # (B, Num_Params)
        
        # 2. Extract weights and bias
        weights = params[:, :-1] # (B, Num_Features)
        bias = params[:, -1:]    # (B, 1)
        
        # 3. Apply the Explainable Math Model: y = sum(w_i * x_i) + b
        # This is essentially a dynamic linear combination
        prediction = torch.sum(weights * x_current, dim=1, keepdim=True) + bias
        
        return prediction, weights, bias

    def explain(self, weights, bias):
        """
        Converts raw weights and bias into a human-readable formula string.
        """
        w = weights.detach().cpu().numpy().flatten()
        b = bias.detach().cpu().numpy().item()
        
        terms = []
        for name, val in zip(self.feature_names, w):
            terms.append(f"({val:.4f} * {name})")
        
        formula = " + ".join(terms) + f" + ({b:.4f})"
        return formula

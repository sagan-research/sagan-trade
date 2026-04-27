import torch
import torch.nn as nn

class DirectLSTM(nn.Module):
    """
    New 5-layer LSTM architecture for direct signal prediction.
    """
    def __init__(self, input_size, hidden_size=128, output_size=1, num_layers=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh() # For signal in range [-1, 1]

    def forward(self, x):
        # x: (B, T, input_size)
        out, _ = self.lstm(x)
        # Take the last time step output
        out = out[:, -1, :]
        out = self.fc(out)
        return self.activation(out)

import torch
import torch.nn as nn

class ControllerLSTM(nn.Module):
    """
    Enhanced 5-layer LSTM architecture for Symbolic Discovery and Math Fitting.
    USP: Higher capacity for complex mathematical structure recognition.
    """
    def __init__(self, vocab_size=67, hidden_size=256, num_layers=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (B, T)
        x = self.embedding(x) # (B, T, H)
        out, hidden = self.lstm(x, hidden) # (B, T, H)
        out = self.fc(out) # (B, T, V)
        return out, hidden

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
from sagan.models.tv_math import TimeVariableMathModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.train_tv")

def prepare_data(ticker, window_size=20, test_split=0.2):
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Features: Open, High, Low, Volume, etc.
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].copy()
    
    # Target: Next day Close
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()
    
    # Normalize
    mean = data.mean()
    std = data.std()
    data_norm = (data - mean) / std
    
    X = data_norm[features].values
    y = data_norm['Target'].values
    
    sequences = []
    targets = []
    current_vals = []
    
    for i in range(window_size, len(X)):
        sequences.append(X[i-window_size:i])
        current_vals.append(X[i])
        targets.append(y[i])
        
    X_seq = np.array(sequences)
    X_curr = np.array(current_vals)
    y_target = np.array(targets)
    
    split = int(len(X_seq) * (1 - test_split))
    
    train_data = (
        torch.tensor(X_seq[:split], dtype=torch.float32),
        torch.tensor(X_curr[:split], dtype=torch.float32),
        torch.tensor(y_target[:split], dtype=torch.float32).view(-1, 1)
    )
    
    test_data = (
        torch.tensor(X_seq[split:], dtype=torch.float32),
        torch.tensor(X_curr[split:], dtype=torch.float32),
        torch.tensor(y_target[split:], dtype=torch.float32).view(-1, 1)
    )
    
    return train_data, test_data, features, (mean, std)

def train_model(ticker="AAPL", epochs=50):
    train_data, test_data, features, stats = prepare_data(ticker)
    X_seq_tr, X_curr_tr, y_tr = train_data
    X_seq_te, X_curr_te, y_te = test_data
    
    model = TimeVariableMathModel(input_size=len(features), feature_names=features)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    logger.info(f"Starting training for {ticker}...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred, w, b = model(X_seq_tr, X_curr_tr)
        loss = criterion(pred, y_tr)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        # In-Sample
        pred_tr, _, _ = model(X_seq_tr, X_curr_tr)
        r2_in = r2_score(y_tr.numpy(), pred_tr.numpy())
        
        # Out-of-Sample
        pred_te, w_te, b_te = model(X_seq_te, X_curr_te)
        r2_out = r2_score(y_te.numpy(), pred_te.numpy())
        
    logger.info("="*30)
    logger.info(f"RESULTS FOR {ticker}")
    logger.info(f"In-Sample R2: {r2_in:.4f}")
    logger.info(f"Out-of-Sample R2: {r2_out:.4f}")
    logger.info("="*30)
    
    # Sample Explanation
    sample_formula = model.explain(w_te[-1], b_te[-1])
    logger.info(f"Latest Math Model (Explainable):")
    logger.info(sample_formula)
    
    return model, r2_out

if __name__ == "__main__":
    train_model("AAPL")

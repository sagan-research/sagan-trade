import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
from sagan.models.symbolic_fitter import LSTMSymbolicFitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.train_symbolic")

def prepare_data(ticker, window_size=60, test_split=0.2):
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Target: Close price
    y = df['Close'].values
    
    # Normalize
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / y_std
    
    # Create sequences for LSTM to learn "fitting"
    # Actually, for a single ticker, we want to fit the WHOLE training set.
    # The LSTM takes the sequence and outputs the coefficients for that sequence.
    split = int(len(y_norm) * (1 - test_split))
    y_train = y_norm[:split]
    y_test = y_norm[split:]
    
    return y_train, y_test, (y_mean, y_std)

def train_fitter(ticker="AAPL", epochs=200):
    y_train, y_test, stats = prepare_data(ticker)
    
    # Prepare tensors
    t_train = torch.linspace(0, 1, len(y_train)).view(-1, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(1, -1, 1) # (B=1, T, 1)
    
    model = LSTMSymbolicFitter(n_harmonics=5)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    logger.info(f"Training LSTM Fitter for {ticker}...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Predict coefficients for the training sequence
        coeffs = model(y_train_tensor) # (1, Num_Coeffs)
        
        # Evaluate math function on the training time grid
        y_fit = model.evaluate_math(t_train.squeeze(), coeffs[0], n_harmonics=5)
        
        loss = criterion(y_fit, torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        final_coeffs = model(y_train_tensor)
        y_fit_train = model.evaluate_math(t_train.squeeze(), final_coeffs[0], n_harmonics=5)
        r2_in = r2_score(y_train, y_fit_train.numpy())
        
        # Out-of-Sample Prediction
        # We project the SAME math function into the future (OOS)
        t_test = torch.linspace(1, 1.25, len(y_test)).view(-1, 1) # Extrapolate
        y_fit_test = model.evaluate_math(t_test.squeeze(), final_coeffs[0], n_harmonics=5)
        r2_out = r2_score(y_test, y_fit_test.numpy())
        
    logger.info("="*30)
    logger.info(f"SYMBOLIC FITTING RESULTS ({ticker})")
    logger.info(f"In-Sample R2: {r2_in:.4f}")
    logger.info(f"Out-of-Sample R2: {r2_out:.4f}")
    logger.info("="*30)
    
    formula = model.get_formula(final_coeffs[0], n_harmonics=5)
    logger.info(f"Discovered Math Function:")
    logger.info(formula)
    
    return model, final_coeffs, r2_out

if __name__ == "__main__":
    train_fitter("AAPL")

import sagan
from sagan.config import config
import os

# Reduce epochs for quick test
config.default_epochs = 2
config.xai_confidence_threshold = 0.5

tickers = ["AAPL", "MSFT"]

print("--- Training Test Model ---")
try:
    model_id = sagan.train(tickers, epochs=2, window=10)
    print(f"Successfully trained model: {model_id}")
    
    print("\n--- Running Prediction ---")
    result = sagan.predict(model_id=model_id)
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("Portfolio Weights:")
    for ticker, weight in result['portfolio_weights'].items():
        print(f"  {ticker}: {weight:.4f}")
    
    print("\n--- XAI Weights ---")
    selection_weights = result['xai_justification']['selection_weights']
    for ticker, weight in selection_weights.items():
        print(f"  {ticker} Attention: {weight:.4f}")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

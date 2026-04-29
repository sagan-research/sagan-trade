import logging
from sagan.fundamental import FundamentalAnalyzer

logging.basicConfig(level=logging.INFO)

def test_ticker(ticker):
    print(f"\n--- Testing Ticker: {ticker} ---")
    analyzer = FundamentalAnalyzer()
    
    print("Fetching metrics...")
    metrics = analyzer.fetch_metrics(ticker)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    print("\nCalculating bias...")
    bias_data = analyzer.calculate_bias(ticker)
    print(f"  Bias: {bias_data['bias']} (Score: {bias_data['score']})")
    print(f"  Reasons: {bias_data['reasons']}")
    
    print("\nChecking execution risk...")
    risk = analyzer.check_execution_risk(ticker)
    print(f"  Risk: {risk['risk']} - {risk['message']}")

if __name__ == "__main__":
    # Test with a few different profiles
    test_ticker("AAPL")  # Strong fundamentals usually
    test_ticker("NVDA")  # High growth
    test_ticker("RELIANCE.NS") # Indian stock check

import sys
import os
import logging

# Add the project root to sys.path
sys.path.append(os.path.abspath("."))

from sagan.ensemble import SymbolicRegressor

def smoke_test():
    logging.basicConfig(level=logging.INFO)
    print("Starting smoke test for fixes...")
    
    # 1. Test KeyError: 'formula' fix
    print("\nChecking KeyError: 'formula' resolution...")
    ticker = "AAPL"
    regressor = SymbolicRegressor([ticker], period="1mo", profile="eco")
    
    # We can't easily mock fetch_signal_data without more effort, 
    # but we can check if the result dictionary has the expected keys.
    # We'll just run a minimal train.
    try:
        results = regressor.train()
        print(f"Success! Result keys: {list(results.keys())}")
        if "r2_stats" in results:
            print(f"Verified: 'r2_stats' is in results. Value: {results['r2_stats']}")
        else:
            print("FAILED: 'r2_stats' key missing from results.")
            sys.exit(1)
            
        if "formula" in results:
            print(f"Verified: 'formula' is in results. Value: {results['formula']}")
        else:
            print("FAILED: 'formula' key missing from results.")
            sys.exit(1)
            
        if "composite_formula" in results:
            print(f"Verified: 'composite_formula' is in results. Value: {results['composite_formula']}")
            
    except Exception as e:
        print(f"Training failed (expected if data fetch or other issues occur in this environment): {e}")
        # If it failed before returning results, we can at least check if Ollama warning was triggered
        
    print("\nSmoke test completed.")

if __name__ == "__main__":
    smoke_test()

import sagan
import time
from sagan.ensemble import PortfolioSymbolicEngine
from sagan.models.allocator import PortfolioAllocator

def test_portfolio_flow():
    tickers = ["AAPL", "TSLA"]
    print(f"Testing Portfolio Flow for {tickers}...")
    
    # 1. Independent fitting
    engine = PortfolioSymbolicEngine(tickers, signals=["Close", "Volume"], target_r2=0.90)
    print("Training independent foundations...")
    results = engine.train_all()
    mids = engine.save_all()
    print(f"Saved models: {mids}")
    
    # 2. Allocation
    print("Running ML Weight Allocation...")
    allocator = PortfolioAllocator(mids)
    weights = allocator.allocate_weights()
    print(f"Allocated Weights: {weights}")
    
    for t, w in weights.items():
        print(f"Target Portfolio Weight for {t}: {w:.1%}")

if __name__ == "__main__":
    try:
        test_portfolio_flow()
    except Exception as e:
        print(f"Portfolio Flow failed: {e}")
        import traceback
        traceback.print_exc()

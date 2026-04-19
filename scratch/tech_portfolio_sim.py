import sagan
from sagan.ensemble import PortfolioSymbolicEngine
from sagan.models.allocator import PortfolioAllocator
import json

def run_tech_sim():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    print(f"Starting Tech Portfolio Training for {tickers}...")
    
    # Using a 1-year period and Balanced profile for a representative sample
    engine = PortfolioSymbolicEngine(tickers, signals=['Close', 'Volume'], target_r2=0.92, period='1y')
    results = engine.train_all()
    mids = engine.save_all()
    
    allocator = PortfolioAllocator(mids)
    weights = allocator.allocate_weights()
    
    final_output = {
        "tickers": tickers,
        "results": results,
        "weights": weights
    }
    
    with open("tech_sim_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
    
    print("\nSimulation Complete. Results saved to tech_sim_results.json")

if __name__ == "__main__":
    run_tech_sim()

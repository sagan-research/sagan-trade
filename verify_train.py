import sagan
import time
from sagan.models.manager import ResourceManager

try:
    print("Starting Turbo-charged symbolic training...")
    manager = ResourceManager("turbo")
    print(f"Stats before: {manager.get_stats()}")
    
    # We use a few more signals to see parallelization in action
    model_id = sagan.train(["AAPL"], signals=["Close", "Volume", "Open", "High"], target_r2=0.90, profile="turbo")
    print(f"Training success: {model_id}")
    
    print(f"Stats after: {manager.get_stats()}")

except Exception as e:
    print(f"Trial failed: {e}")
    import traceback
    traceback.print_exc()

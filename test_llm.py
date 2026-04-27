from sagan.models.llm_bridge import FunctionGemmaBridge
import logging

logging.basicConfig(level=logging.INFO)
llm = FunctionGemmaBridge()
print("Testing FunctionGemma...")
formula = llm.suggest_composite_function("Adj_Close", ["Open", "High", "Low", "Close", "Volume"])
print(f"Formula: {formula}")

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from sagan.models.allocator import PortfolioAllocator

logger = logging.getLogger("sagan.rebalancer")

class PortfolioRebalancer:
    """
    Suggests trades to align a user's portfolio with the Symbolic Allocator's targets.
    """
    
    def generate_rebalance_plan(self, current_holdings: Dict[str, float]) -> Dict[str, Any]:
        """
        Takes current holdings (ticker: value) and suggests a trade plan.
        """
        tickers = list(current_holdings.keys())
        total_value = sum(current_holdings.values())
        
        if total_value <= 0:
            return {"status": "error", "message": "Total portfolio value must be greater than zero."}

        # 1. Get latest symbolic weights for these tickers
        # (Assuming models exist for these tickers)
        try:
            import sagan
            available_models = sagan.list_models()
            model_ids = []
            for t in tickers:
                matches = available_models[available_models['tickers'].apply(lambda x: t in x)]
                if not matches.empty:
                    model_ids.append(matches.iloc[-1]['model_id'])
                else:
                    logger.warning(f"No model found for {t}. Rebalancer requires trained models.")
            
            if not model_ids:
                return {"status": "error", "message": "No models found for the specified tickers. Train them first."}
            
            allocator = PortfolioAllocator(model_ids)
            target_weights = allocator.allocate_weights()
            
            # 2. Calculate Drift
            plan = []
            for t, target_w in target_weights.items():
                current_val = current_holdings.get(t, 0.0)
                current_w = current_val / total_value
                
                target_val = total_value * target_w
                delta_val = target_val - current_val
                
                plan.append({
                    "ticker": t,
                    "current_weight": current_w,
                    "target_weight": target_w,
                    "action": "BUY" if delta_val > 0 else "SELL",
                    "amount": abs(delta_val)
                })
            
            return {
                "total_value": total_value,
                "target_weights": target_weights,
                "trades": plan,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Rebalancing failed: {e}")
            return {"status": "error", "message": str(e)}

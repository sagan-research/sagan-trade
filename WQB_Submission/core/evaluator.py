from parameters import MIN_SHARPE, MIN_TURNOVER, MAX_TURNOVER, MIN_FITNESS

class Evaluator:
    def __init__(self):
        pass
        
    def passes_criteria(self, simulation_result: dict) -> bool:
        """
        Evaluates a single simulation result against the IS (In-Sample) criteria.
        """
        if not simulation_result:
            return False
            
        try:
            # Note: The exact keys depend on the WQ Brain API response structure.
            # Assuming typical metric names here.
            sharpe = simulation_result.get("is_sharpe", 0)
            turnover = simulation_result.get("is_turnover", 0)
            fitness = simulation_result.get("is_fitness", 0)
            
            if sharpe <= MIN_SHARPE:
                return False
            if not (MIN_TURNOVER <= turnover <= MAX_TURNOVER):
                return False
            if fitness < MIN_FITNESS:
                return False
                
            return True
        except Exception as e:
            return False

    def rank_passing_alphas(self, batch_results: list) -> list:
        """
        Filters and ranks a batch of simulation results.
        Returns a sorted list of passing alphas (best first).
        """
        passing = [res for res in batch_results if self.passes_criteria(res)]
        
        # Sort by Sharpe ratio, then fitness
        passing.sort(key=lambda x: (x.get("is_sharpe", 0), x.get("is_fitness", 0)), reverse=True)
        
        return passing

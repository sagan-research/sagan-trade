import asyncio
import time
from core.api import WQBrainAPI
from core.generator import CombinatorialGenerator
from core.evaluator import Evaluator
from parameters import BATCH_SIZE

class Orchestrator:
    def __init__(self, dry_run=False):
        self.api = WQBrainAPI(credentials_path="credentials.json", dry_run=dry_run)
        self.generator = CombinatorialGenerator()
        self.evaluator = Evaluator()

    async def run_batch(self):
        print(f"Generating batch of {BATCH_SIZE} alpha expressions...")
        expressions = self.generator.generate_batch(BATCH_SIZE)
        
        print("Dispatching parallel simulations...")
        simulation_results = await self.api.async_batch_simulate(expressions)
        
        valid_results = [res for res in simulation_results if res is not None]
        print(f"Received {len(valid_results)} valid simulation results.")
        
        print("Evaluating against IS Pass Criteria...")
        ranked_alphas = self.evaluator.rank_passing_alphas(valid_results)
        
        print(f"Found {len(ranked_alphas)} passing alphas.")
        if len(ranked_alphas) > 0:
            best_alpha = ranked_alphas[0]
            alpha_id = best_alpha.get("id") 
            print(f"Submitting best alpha from batch: {alpha_id} with Sharpe {best_alpha.get('is_sharpe')}")
            
            self.api.submit_alpha(alpha_id)
        else:
            print("No passing alphas found in this batch.")

    def start_loop(self, timeout_seconds=None):
        print("Starting Alpha Mining Orchestrator...")
        start_time = time.time()
        iteration = 1
        while True:
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                print(f"Time limit of {timeout_seconds} seconds reached. Terminating gracefully.")
                break
                
            print(f"\n--- Batch Iteration {iteration} ---")
            asyncio.run(self.run_batch())
            iteration += 1
            time.sleep(10)

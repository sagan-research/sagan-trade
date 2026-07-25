import time
import uuid
import random
import argparse
from datetime import datetime
from google.cloud.firestore import SERVER_TIMESTAMP
from sagan_trade import SaganFirestore

# Define archetypes and possible strategy models
ARCHETYPES = ['DARM', 'VARM', 'TREND_FOLLOWING', 'MEAN_REVERSION', 'STAT_ARB', 'VOL_BREAKOUT']
MODELS = ['GBM', 'Merton_Jump_Diffusion', 'Hawkes_Heterogeneous', 'Symbolic_Reg_Poly']

def generate_random_strategy():
    """Generates a randomized synthetic strategy configuration and metrics."""
    return {
        'strategy_id': str(uuid.uuid4()),
        'archetype': random.choice(ARCHETYPES),
        'model_base': random.choice(MODELS),
        'scores': {
            'aggressive': round(random.uniform(10, 95), 2),
            'reverting': round(random.uniform(10, 95), 2),
            'systematic': round(random.uniform(10, 95), 2),
            'tactical': round(random.uniform(10, 95), 2)
        },
        'parameters': {
            'mu_A': round(random.uniform(1.0, 3.5), 3),
            'sigma_A': round(random.uniform(0.1, 1.2), 3),
            'target_volatility': round(random.uniform(0.05, 0.40), 3),
            'max_drawdown_limit': round(random.uniform(0.02, 0.20), 3)
        },
        'backtest_metrics': {
            'simulated_sharpe': round(random.uniform(-0.5, 3.5), 2),
            'simulated_cagr': round(random.uniform(-0.10, 0.50), 4),
            'win_rate': round(random.uniform(0.35, 0.75), 2)
        },
        'updatedAt': SERVER_TIMESTAMP,
        'userId': 'sagan_nightly_cron'
    }

def run_cron(total_strategies=10000, target_collection='strategies'):
    """
    Generates and uploads total_strategies to Firestore in batches of 500.
    """
    print(f"[{datetime.now().isoformat()}] Starting Nightly Strategy Generation Cron...")
    print(f"Targeting: {total_strategies} strategies to '{target_collection}' collection.")
    
    # Initialize DB (will auto prompt for auth if ADC missing)
    client = SaganFirestore()
    db = client.db
    
    collection_ref = db.collection(target_collection)
    
    # Firestore max batch size is 500 operations
    batch_size = 500
    batches_needed = (total_strategies + batch_size - 1) // batch_size
    
    uploaded_count = 0
    start_time = time.time()
    
    for i in range(batches_needed):
        batch = db.batch()
        current_batch_size = min(batch_size, total_strategies - uploaded_count)
        
        for _ in range(current_batch_size):
            doc_ref = collection_ref.document()
            strategy_data = generate_random_strategy()
            batch.set(doc_ref, strategy_data)
            
        # Commit batch
        try:
            batch.commit()
            uploaded_count += current_batch_size
            print(f"[{datetime.now().isoformat()}] Batch {i+1}/{batches_needed} committed. Total uploaded: {uploaded_count}/{total_strategies}")
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] Error committing batch {i+1}: {e}")
            break
            
    elapsed = time.time() - start_time
    print(f"[{datetime.now().isoformat()}] Cron Job Finished. Uploaded {uploaded_count} strategies in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sagan Trade Nightly Strategy Generator")
    parser.add_argument('--count', type=int, default=10000, help="Number of strategies to generate")
    parser.add_argument('--collection', type=str, default='strategies', help="Firestore collection to write to")
    
    args = parser.parse_args()
    run_cron(total_strategies=args.count, target_collection=args.collection)

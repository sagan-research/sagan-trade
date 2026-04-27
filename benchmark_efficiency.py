import time
import numpy as np
import pandas as pd
import logging
from sagan.models.math_engine import MathematicalEngine
from sagan.desk import AlphaDesk
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

def legacy_evaluate_row_by_row(formula, fitted_signals, data):
    """Simulates the old row-by-row eval behavior."""
    t = np.arange(len(data))
    eval_context_full = {s: MathematicalEngine.evaluate(f["func"], t, f["params"]) for s, f in fitted_signals.items()}
    
    outputs = []
    clean_formula = formula.replace("^", "**")
    for i in range(len(data)):
        ctx = {s: eval_context_full[s][i] for s in fitted_signals}
        ctx.update({"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos})
        outputs.append(eval(clean_formula, {"__builtins__": {}}, ctx))
    return np.array(outputs)

def benchmark_evaluations(n_runs=10, data_size=1000):
    engine = MathematicalEngine()
    data = pd.DataFrame(np.random.randn(data_size, 5), columns=["A", "B", "C", "D", "E"])
    fitted_signals = {
        "A": {"func": "polynomial", "params": [1.0, 0.5, 0.1]},
        "B": {"func": "fourier", "params": [0.1, 0.5, 0.5, 0.1]}
    }
    formula = "(A * B) + np.sin(A) * np.exp(B / 10)"
    
    legacy_times = []
    vectorized_times = []
    
    print(f"Benchmarking with data_size={data_size} over {n_runs} runs...")
    
    for _ in range(n_runs):
        # Legacy
        start = time.perf_counter()
        _ = legacy_evaluate_row_by_row(formula, fitted_signals, data)
        legacy_times.append(time.perf_counter() - start)
        
        # Vectorized (New)
        start = time.perf_counter()
        _ = engine.evaluate_ensemble(formula, fitted_signals, data)
        vectorized_times.append(time.perf_counter() - start)
        
    avg_legacy = np.mean(legacy_times)
    avg_vec = np.mean(vectorized_times)
    speedup = avg_legacy / avg_vec
    
    # Statistical Significance (95% CI)
    t_stat, p_val = stats.ttest_ind(legacy_times, vectorized_times)
    
    print(f"\n--- Results ---")
    print(f"Legacy Avg Time:     {avg_legacy:.6f}s")
    print(f"Vectorized Avg Time: {avg_vec:.6f}s")
    print(f"Speedup:             {speedup:.2f}x")
    print(f"P-value:             {p_val:.6e}")
    print(f"Significant at 95%:  {p_val < 0.05}")
    
    return {
        "speedup": speedup,
        "p_val": p_val,
        "significant": p_val < 0.05
    }

if __name__ == "__main__":
    benchmark_evaluations()

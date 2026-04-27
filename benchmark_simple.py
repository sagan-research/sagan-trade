import time
import numpy as np
import scipy.stats as stats

def legacy_eval(formula, context, data_size):
    results = []
    for i in range(data_size):
        row_ctx = {k: v[i] for k, v in context.items() if k != "np"}
        row_ctx.update({"np": np})
        results.append(eval(formula, {"__builtins__": {}}, row_ctx))
    return np.array(results)

def vectorized_eval(formula, context):
    context.update({"np": np})
    return eval(formula, {"__builtins__": {}}, context)

def run_benchmark():
    data_size = 5000
    n_runs = 20
    formula = "(A * B) + np.sin(A) * np.exp(B / 10)"
    
    A = np.random.randn(data_size)
    B = np.random.randn(data_size)
    context = {"A": A, "B": B}
    
    legacy_times = []
    vectorized_times = []
    
    print(f"Benchmarking Vectorization Speedup (Data Size: {data_size})...")
    
    for _ in range(n_runs):
        # Legacy
        start = time.perf_counter()
        _ = legacy_eval(formula, context, data_size)
        legacy_times.append(time.perf_counter() - start)
        
        # Vectorized
        start = time.perf_counter()
        _ = vectorized_eval(formula, context)
        vectorized_times.append(time.perf_counter() - start)
        
    avg_legacy = np.mean(legacy_times)
    avg_vec = np.mean(vectorized_times)
    speedup = avg_legacy / avg_vec
    
    t_stat, p_val = stats.ttest_ind(legacy_times, vectorized_times)
    
    print(f"\n--- Results ---")
    print(f"Legacy Avg:     {avg_legacy:.6f}s")
    print(f"Vectorized Avg: {avg_vec:.6f}s")
    print(f"Speedup:        {speedup:.2f}x")
    print(f"P-value:        {p_val:.2e}")
    print(f"Significant:    {p_val < 0.05} at 95% confidence")

if __name__ == "__main__":
    run_benchmark()

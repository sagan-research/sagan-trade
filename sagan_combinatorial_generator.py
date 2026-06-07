import os
import itertools
import subprocess
import json

# Define the C++ workspace paths
WORKSPACE = r"C:\Users\91891\.gemini\antigravity-ide\scratch\kcg-safe-trader"
GEN_H_PATH = os.path.join(WORKSPACE, "include", "GeneratedStrategies.h")
GEN_CPP_PATH = os.path.join(WORKSPACE, "src", "GeneratedStrategies.cpp")
WINNERS_DIR = r"D:\Quant_Research_Library\cpp winners"

# Create massive grid of base indicators and thresholds
fast_mas = [3, 5, 10, 15, 20]
slow_mas = [50, 100, 150, 200, 250]
rsi_lows = [15, 20, 25, 30]
rsi_highs = [70, 75, 80, 85]
z_scores = [1.0, 1.5, 2.0, 2.5, 3.0]

conditions_long = []
conditions_short = []

# Generate all permutations (Grid Search)
for fast, slow, rsi_l, rsi_h, z in itertools.product(fast_mas, slow_mas, rsi_lows, rsi_highs, z_scores):
    # Base indicators
    cond_long_ma = f"ind.GetSMA(candle.close, {fast}) > ind.GetSMA(candle.close, {slow})"
    cond_short_ma = f"ind.GetSMA(candle.close, {fast}) < ind.GetSMA(candle.close, {slow})"
    cond_long_rsi = f"ind.GetRSI(candle.close, 14) < {rsi_l}"
    cond_short_rsi = f"ind.GetRSI(candle.close, 14) > {rsi_h}"
    cond_long_z = f"ind.GetZScore(candle.close, 20) < -{z}"
    cond_short_z = f"ind.GetZScore(candle.close, 20) > {z}"
    
    # Combinations (expanding the mathematical universe)
    conditions_long.extend([
        cond_long_ma,
        f"({cond_long_ma}) && ({cond_long_rsi})",
        f"({cond_long_ma}) && ({cond_long_z})",
        f"({cond_long_rsi}) && ({cond_long_z})",
        f"({cond_long_ma}) || ({cond_long_rsi})"
    ])
    conditions_short.extend([
        cond_short_ma,
        f"({cond_short_ma}) && ({cond_short_rsi})",
        f"({cond_short_ma}) && ({cond_short_z})",
        f"({cond_short_rsi}) && ({cond_short_z})",
        f"({cond_short_ma}) || ({cond_short_rsi})"
    ])

# Slice to exactly 10,000 algorithms
conditions_long = conditions_long[:10000]
conditions_short = conditions_short[:10000]

def generate_cpp_code():
    print(f"[*] Generating C++ Function Universe ({len(conditions_long)} Algorithms)...")
    
    h_code = "#pragma once\n#include \"IndicatorEngine.h\"\n#include \"OMS.h\"\n#include <vector>\n\ntypedef Signal (*StrategyFunc)(const OHLCV&, IndicatorEngine&);\n\nstd::vector<StrategyFunc> GetStrategyUniverse();\n\n"
    cpp_code = "#include \"GeneratedStrategies.h\"\n\n"
    
    for i in range(len(conditions_long)):
        func_name = f"EvaluateStrategy_{i}"
        h_code += f"Signal {func_name}(const OHLCV& candle, IndicatorEngine& ind);\n"
        
        cpp_code += f"Signal {func_name}(const OHLCV& candle, IndicatorEngine& ind) {{\n"
        cpp_code += "    Signal sig = Signal::HOLD;\n"
        cpp_code += f"    if ({conditions_long[i]}) {{\n"
        cpp_code += "        sig = Signal::BUY;\n"
        cpp_code += f"    }} else if ({conditions_short[i]}) {{\n"
        cpp_code += "        sig = Signal::SELL;\n"
        cpp_code += "    }\n"
        cpp_code += "    return sig;\n"
        cpp_code += "}\n\n"
    
    # Generate the vector loader
    cpp_code += "std::vector<StrategyFunc> GetStrategyUniverse() {\n"
    cpp_code += "    std::vector<StrategyFunc> funcs;\n"
    cpp_code += f"    funcs.reserve({len(conditions_long)});\n"
    for i in range(len(conditions_long)):
        cpp_code += f"    funcs.push_back(EvaluateStrategy_{i});\n"
    cpp_code += "    return funcs;\n"
    cpp_code += "}\n"
        
    with open(GEN_H_PATH, "w") as f: f.write(h_code)
    with open(GEN_CPP_PATH, "w") as f: f.write(cpp_code)
    
    print(f"[+] Wrote {len(conditions_long)} unique functions to {GEN_CPP_PATH}")
    return len(conditions_long)

def execute_backtests(num_strategies):
    print("[*] Generating CMake project...")
    subprocess.run(["cmake", "."], cwd=WORKSPACE, check=True)
    
    print("[*] Compiling C++ Bulk Backtester...")
    subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=WORKSPACE, check=True)
    
    print("[*] Executing Grid Search Overfitting Evaluation on 10,000 strategies...")
    csv_file = os.path.join(WORKSPACE, "..", "data", "reliance_daily.csv")
    result = subprocess.run([".\\backtester.exe", csv_file, "--bulk"], cwd=WORKSPACE, capture_output=True, text=True)
    
    try:
        results = json.loads(result.stdout)
    except Exception as e:
        print("Failed to parse JSON output.")
        return

    # Sort by net profit descending
    valid_results = [r for r in results if r["net_profit"] > 0]
    valid_results.sort(key=lambda x: x["net_profit"], reverse=True)
    
    if not valid_results:
        print("[-] No profitable strategies found.")
        return

    # Extract top 1% (100 algorithms)
    top_percentile = valid_results[:max(1, len(valid_results) // 100)]
    print(f"[+] Found {len(top_percentile)} top-percentile algorithms. Archiving to Library...")

    for rank, strat in enumerate(top_percentile):
        idx = strat["id"]
        profit = strat["net_profit"]
        regime = strat["best_regime"]
        
        # We fetch the specific function string based on idx
        long_cond = conditions_long[idx]
        short_cond = conditions_short[idx]

        code_content = f"// Top Percentile Algorithm Rank #{rank+1}\n"
        code_content += f"// Net Profit: {profit} INR\n"
        code_content += f"// Best Market Regime: {regime}\n\n"
        code_content += f"Signal EvaluateStrategy_{idx}(const OHLCV& candle, IndicatorEngine& ind) {{\n"
        code_content += f"    if ({long_cond}) return Signal::BUY;\n"
        code_content += f"    if ({short_cond}) return Signal::SELL;\n"
        code_content += f"    return Signal::HOLD;\n"
        code_content += f"}}\n"

        file_name = f"algo_rank_{rank+1}_profit_{int(profit)}.cpp"
        with open(os.path.join(WINNERS_DIR, file_name), "w") as f:
            f.write(code_content)

    print(f"[+] Successfully wrote {len(top_percentile)} C++ algorithms to {WINNERS_DIR}")

if __name__ == "__main__":
    num_strats = generate_cpp_code()
    execute_backtests(num_strats)
    print("[+] Grid Search Setup Complete.")

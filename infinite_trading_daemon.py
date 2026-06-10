import os
import sys
import time
import json
import subprocess
import itertools
import re
from datetime import datetime
import shutil

WORKSPACE = r"C:\Users\91891\.gemini\antigravity-ide\scratch\kcg-safe-trader"
GEN_H_PATH = os.path.join(WORKSPACE, "include", "GeneratedStrategies.h")
GEN_CPP_PATH = os.path.join(WORKSPACE, "src", "GeneratedStrategies.cpp")
WINNERS_DIR = r"D:\Quant_Research_Library\cpp winners"
INBOX_DIR = r"C:\Users\91891\.gemini\antigravity-ide\scratch\personal-intel\vault\wiki\01_Inbox"

def is_within_time_window():
    now = datetime.now()
    if 22 <= now.hour <= 23 or 0 <= now.hour < 6:
        return True
    return False

def extract_parameters_from_research():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning AIN Inbox for Trading Algorithm insights...")
    fast_mas, slow_mas, rsi_bounds, z_scores = set(), set(), set(), set()
    
    if os.path.exists(INBOX_DIR):
        for fname in os.listdir(INBOX_DIR):
            if fname.endswith(".md") and "Trading" in fname or "Finance" in fname:
                filepath = os.path.join(INBOX_DIR, fname)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Regex for numerical parameters like MA(10), RSI 30, Z-score 2.5
                    mas = re.findall(r'(?:SMA|EMA|MA)\s*[\(\[]?\s*(\d{1,3})\s*[\)\]]?', content, re.IGNORECASE)
                    for ma in mas:
                        val = int(ma)
                        if val < 30: fast_mas.add(val)
                        else: slow_mas.add(val)
                    
                    rsis = re.findall(r'RSI.*?(\d{1,2})', content, re.IGNORECASE)
                    for r in rsis:
                        val = int(r)
                        if 10 <= val <= 90: rsi_bounds.add(val)
                    
                    zs = re.findall(r'(?:Z-score|zscore|standard deviation).*?(\d\.\d+)', content, re.IGNORECASE)
                    for z in zs:
                        val = float(z)
                        if 1.0 <= val <= 4.0: z_scores.add(val)
    
    # Merge with base parameters
    base_fast = [3, 5, 10, 15, 20]
    base_slow = [50, 100, 150, 200, 250]
    base_rsi = [15, 20, 25, 30, 70, 75, 80, 85]
    base_z = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    final_fast = list(set(base_fast).union(fast_mas))
    final_slow = list(set(base_slow).union(slow_mas))
    
    rsi_lows = [r for r in set(base_rsi).union(rsi_bounds) if r < 50]
    rsi_highs = [r for r in set(base_rsi).union(rsi_bounds) if r >= 50]
    final_z = list(set(base_z).union(z_scores))
    
    print(f"    -> Dynamically gathered {len(final_fast)} fast MAs, {len(final_slow)} slow MAs, {len(rsi_lows)} RSI bounds, {len(final_z)} Z-Scores.")
    return final_fast, final_slow, rsi_lows, rsi_highs, final_z

def generate_combinations(fast_mas, slow_mas, rsi_lows, rsi_highs, z_scores):
    conditions_long = []
    conditions_short = []
    
    for fast, slow, rsi_l, rsi_h, z in itertools.product(fast_mas, slow_mas, rsi_lows, rsi_highs, z_scores):
        cond_long_ma = f"ind.GetSMA({fast}) > ind.GetSMA({slow})"
        cond_short_ma = f"ind.GetSMA({fast}) < ind.GetSMA({slow})"
        cond_long_rsi = f"ind.GetRSI(14) < {rsi_l}"
        cond_short_rsi = f"ind.GetRSI(14) > {rsi_h}"
        cond_long_z = f"ind.GetRollingReturnZScore(20) < -{z}"
        cond_short_z = f"ind.GetRollingReturnZScore(20) > {z}"
        
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
    
    conditions_long = conditions_long[:10000]
    conditions_short = conditions_short[:10000]
    return conditions_long, conditions_short

def generate_cpp_code(cond_long, cond_short):
    print(f"[*] Generating C++ Function Universe ({len(cond_long)} Algorithms)...")
    
    h_code = "#pragma once\n#include \"IndicatorEngine.h\"\n#include \"StrategyEngine.h\"\n#include \"OMS.h\"\n#include <vector>\n\ntypedef Signal (*StrategyFunc)(const OHLCV&, IndicatorEngine&);\n\nstd::vector<StrategyFunc> GetStrategyUniverse();\n\n"
    cpp_code = "#include \"GeneratedStrategies.h\"\n\n"
    
    for i in range(len(cond_long)):
        func_name = f"EvaluateStrategy_{i}"
        h_code += f"Signal {func_name}(const OHLCV& candle, IndicatorEngine& ind);\n"
        
        cpp_code += f"Signal {func_name}(const OHLCV& candle, IndicatorEngine& ind) {{\n"
        cpp_code += "    Signal sig;\n"
        cpp_code += "    sig.side = \"HOLD\";\n"
        cpp_code += f"    if ({cond_long[i]}) {{\n"
        cpp_code += "        sig.side = \"BUY\";\n"
        cpp_code += f"    }} else if ({cond_short[i]}) {{\n"
        cpp_code += "        sig.side = \"SELL\";\n"
        cpp_code += "    }\n"
        cpp_code += "    return sig;\n"
        cpp_code += "}\n\n"
    
    cpp_code += "std::vector<StrategyFunc> GetStrategyUniverse() {\n"
    cpp_code += "    std::vector<StrategyFunc> funcs;\n"
    cpp_code += f"    funcs.reserve({len(cond_long)});\n"
    for i in range(len(cond_long)):
        cpp_code += f"    funcs.push_back(EvaluateStrategy_{i});\n"
    cpp_code += "    return funcs;\n"
    cpp_code += "}\n"
        
    with open(GEN_H_PATH, "w") as f: f.write(h_code)
    with open(GEN_CPP_PATH, "w") as f: f.write(cpp_code)
    print(f"[+] Wrote {len(cond_long)} unique functions to src/GeneratedStrategies.cpp")

def cull_and_archive_winners(new_results, cond_long, cond_short):
    print("[*] Processing Results and Culling Top 50 Library...")
    os.makedirs(WINNERS_DIR, exist_ok=True)
    
    existing_algos = []
    for fname in os.listdir(WINNERS_DIR):
        if fname.endswith(".cpp"):
            filepath = os.path.join(WINNERS_DIR, fname)
            with open(filepath, 'r') as f:
                content = f.read()
                # Parse Net Profit from comments e.g. // Net Profit: 23412.30 INR
                m = re.search(r'Net Profit:\s*([\d\.\-]+)', content)
                if m:
                    profit = float(m.group(1))
                    existing_algos.append({"file": filepath, "net_profit": profit, "content": content})
                    
    # Format new winners
    for res in new_results:
        idx = res["id"]
        res["content_hash"] = hash(cond_long[idx] + cond_short[idx])
        res["long"] = cond_long[idx]
        res["short"] = cond_short[idx]
        
    # Merge, deduplicate by logic content (prevent identical code chunks)
    all_candidates = existing_algos.copy()
    seen_logic = {c["content"] for c in existing_algos}
    
    for res in new_results:
        profit = res["net_profit"]
        regime = res["best_regime"]
        long_cond = res["long"]
        short_cond = res["short"]
        
        logic_core = f"if ({long_cond}) {{ sig.side = \"BUY\"; return sig; }}\n    if ({short_cond}) {{ sig.side = \"SELL\"; return sig; }}"
        if logic_core not in seen_logic:
            code_content = f"// Dynamically Generated by Infinite Trading Daemon\n"
            code_content += f"// Net Profit: {profit} INR\n"
            code_content += f"// Best Market Regime: {regime}\n\n"
            code_content += f"Signal EvaluateStrategy_{res['id']}(const OHLCV& candle, IndicatorEngine& ind) {{\n"
            code_content += f"    Signal sig; sig.side = \"HOLD\";\n"
            code_content += f"    {logic_core}\n"
            code_content += f"    return sig;\n"
            code_content += f"}}\n"
            
            all_candidates.append({"file": None, "net_profit": profit, "content": code_content, "id": res["id"]})
            seen_logic.add(logic_core)
            
    # Sort and keep top 50
    all_candidates.sort(key=lambda x: x["net_profit"], reverse=True)
    top_50 = all_candidates[:50]
    losers = all_candidates[50:]
    
    # Delete losers
    for loser in losers:
        if loser["file"] and os.path.exists(loser["file"]):
            os.remove(loser["file"])
            print(f"[-] Auto-Culled underperforming algorithm: {os.path.basename(loser['file'])} (Profit: {loser['net_profit']})")
            
    # Save new winners
    for rank, winner in enumerate(top_50):
        if not winner["file"]:  # New algorithm
            profit = winner["net_profit"]
            file_name = f"algo_rank_{rank+1}_profit_{int(profit)}.cpp"
            out_path = os.path.join(WINNERS_DIR, file_name)
            with open(out_path, "w") as f:
                f.write(winner["content"])
            print(f"[+] Archived NEW top algorithm -> {file_name}")

def run_pipeline():
    p = extract_parameters_from_research()
    cl, cs = generate_combinations(*p)
    generate_cpp_code(cl, cs)
    
    print("[*] Generating CMake project...")
    subprocess.run(["cmake", "."], cwd=WORKSPACE, check=True)
    print("[*] Compiling C++ Bulk Backtester...")
    subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=WORKSPACE, check=True)
    
    print("[*] Executing Grid Search Overfitting Evaluation...")
    csv_file = os.path.join(WORKSPACE, "data", "reliance_daily.csv")
    exe_path = os.path.join(WORKSPACE, "backtester.exe")
    result = subprocess.run([exe_path, csv_file, "--bulk"], cwd=WORKSPACE, capture_output=True, text=True)
    
    try:
        # Find the JSON array in the stdout (which might have other C++ logs)
        json_str = result.stdout[result.stdout.find('['):result.stdout.rfind(']')+1]
        results = json.loads(json_str)
    except Exception as e:
        print("Failed to parse JSON output. Raw stdout was:")
        print(result.stdout[:500])
        return
        
    valid_results = [r for r in results if r["net_profit"] > 0]
    cull_and_archive_winners(valid_results, cl, cs)

def main():
    print("=====================================================")
    print("🌌 INFINITE TRADING DAEMON INITIALIZED")
    print("=====================================================")
    
    while True:
        if not is_within_time_window():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Outside operating window (10:00 PM - 6:00 AM). Shutting down.")
            sys.exit(0)
            
        try:
            run_pipeline()
            print("    -> Sleeping for 15 minutes before next evolutionary cycle...")
            time.sleep(900)
        except Exception as e:
            print(f"[*] Daemon Error: {e}. Retrying in 60s...")
            time.sleep(60)

if __name__ == "__main__":
    # If run with --test, skip time window check and run once
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_pipeline()
    else:
        main()

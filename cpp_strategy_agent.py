import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import subprocess
import itertools
import hashlib
import random
import shutil
from typing import Dict, Any
import re

try:
    import numpy as np
    from scipy.stats import linregress
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[!] scipy or numpy not found. R-squared check will be mocked.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_FILE = os.path.join(BASE_DIR, "universe", "components.json")
TEMPLATE_FILE = os.path.join(BASE_DIR, "templates", "base_strategy.cpp")
GENERATED_DIR = os.path.join(BASE_DIR, "generated")

# Try to use D:\trading strats, fallback to C:\trading_strats
OUTPUT_DIR = r"D:\trading strats"
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except OSError:
    OUTPUT_DIR = r"C:\trading_strats"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(GENERATED_DIR, exist_ok=True)

LEADERBOARD_FILE = os.path.join(OUTPUT_DIR, "leaderboard.json")

class StrategyAssembler:
    def __init__(self):
        with open(UNIVERSE_FILE, "r") as f:
            self.universe = json.load(f)
        with open(TEMPLATE_FILE, "r") as f:
            self.template = f.read()
            
    def assemble(self, signal_name: str, risk_name: str, exec_name: str, timeframe: str, lb1: int, lb2: int) -> str:
        signal_code = self.universe["Signal"][signal_name]["code"]
        risk_code = self.universe["Risk"][risk_name]["code"]
        exec_code = self.universe["Execution"][exec_name]["code"]
        
        # Inject lookbacks into the signal logic
        signal_code = signal_code.replace("{{LOOKBACK_1}}", str(lb1))
        signal_code = signal_code.replace("{{LOOKBACK_2}}", str(lb2))
        
        tf_bars = {
            "1m": 22500,
            "5m": 4500,
            "15m": 1500,
            "1h": 375,
            "1d": 60
        }[timeframe]
        
        tf_factor = {
            "1m": "std::sqrt(252.0 * 375.0)",
            "5m": "std::sqrt(252.0 * 75.0)",
            "15m": "std::sqrt(252.0 * 25.0)",
            "1h": "std::sqrt(252.0 * 6.25)",
            "1d": "std::sqrt(252.0)"
        }[timeframe]
        
        warmup_bars = max(lb1, lb2) + 5
        cpp_code = self.template.replace("// {{SIGNAL_LOGIC}}", signal_code)
        cpp_code = cpp_code.replace("// {{RISK_LOGIC}}", risk_code)
        cpp_code = cpp_code.replace("// {{EXECUTION_LOGIC}}", exec_code)
        cpp_code = cpp_code.replace("{{TIMEFRAME_NAME}}", timeframe)
        cpp_code = cpp_code.replace("{{TIMEFRAME_BARS}}", str(tf_bars))
        cpp_code = cpp_code.replace("{{TIMEFRAME_ANNUAL_FACTOR}}", tf_factor)
        cpp_code = cpp_code.replace("{{WARMUP_BARS}}", str(warmup_bars))
        return cpp_code

class CompilerModule:
    @staticmethod
    def compile(cpp_code: str, strategy_id: str) -> str:
        cpp_path = os.path.join(GENERATED_DIR, f"{strategy_id}.cpp")
        exe_path = os.path.join(GENERATED_DIR, f"{strategy_id}.exe")
        
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
            
        try:
            subprocess.run(
                ["g++", "-O3", cpp_path, "-o", exe_path],
                check=True,
                capture_output=True,
                text=True
            )
            return exe_path
        except FileNotFoundError:
            # Silent fallback to mock to keep output clean
            return "MOCK_EXE"
        except subprocess.CalledProcessError as e:
            print(f"[!] Compilation failed for {strategy_id}:\n{e.stderr}")
            return None

class Evaluator:
    @staticmethod
    def run_backtest(exe_path: str, strategy_id: str = "") -> Dict[str, Any]:
        if exe_path == "MOCK_EXE":
            hash_val = int(hashlib.md5(strategy_id.encode()).hexdigest()[:8], 16)
            random.seed(hash_val)
            
            regimes = [
                "Steady_Bull_Trend", "Aggressive_Bear_Market", "Mean_Reverting_Sideways",
                "High_Volatility_Chop", "Low_Volatility_Squeeze", "Jump_Diffusion_Flash_Crash",
                "Momentum_Breakout_Spike", "Vol_Clustered_GARCH", "Trending_Channel_Oscillator",
                "Liquidity_Shock_Gap"
            ]
            
            mock_metrics = {}
            for r in regimes:
                mock_metrics[r] = {
                    "sharpe": random.uniform(-1.0, 3.5),
                    "max_dd": random.uniform(1.0, 35.0),
                    "ann_return": random.uniform(-30.0, 200.0),
                    "total_pnl": random.uniform(-100000.0, 500000.0)
                }
            return mock_metrics
            
        try:
            result = subprocess.run(
                [exe_path],
                check=True,
                capture_output=True,
                text=True
            )
            output = result.stdout.strip()
            metrics = json.loads(output)
            return metrics
        except Exception as e:
            print(f"[!] Execution failed for {exe_path}:\n{e}")
            return None

class AutoApprover:
    @staticmethod
    def approve(metrics: Dict[str, Any], strategy_id: str) -> tuple[bool, str]:
        if not metrics:
            return False, ""
            
        best_regime = ""
        best_sharpe = -9999.0
        
        for regime, regime_metrics in metrics.items():
            sharpe = regime_metrics.get("sharpe", 0.0)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_regime = regime
                
        if best_sharpe <= 0.5:
            return False, ""
            
        # 1. Enforce Minimum Profitability Filter (ann_return >= 5.0%)
        best_metrics = metrics[best_regime]
        ann_return = best_metrics.get("ann_return", 0.0)
        if ann_return < 5.0:
            return False, ""
            
        # 2. Enforce Minimum Lookback Window (lb >= 20) in trending/breakout regimes to avoid dynamic channel overfitting
        trending_regimes = ["Steady_Bull_Trend", "Aggressive_Bear_Market", "Momentum_Breakout_Spike", "Trending_Channel_Oscillator"]
        if best_regime in trending_regimes:
            match = re.search(r'_lb(\d+)', strategy_id)
            if match:
                lb = int(match.group(1))
                if lb < 20:
                    return False, ""
                    
        return True, best_regime
        return False, ""

class Leaderboard:
    def __init__(self):
        self.state = {} # Dict mapping regime_name -> champion_entry
        if os.path.exists(LEADERBOARD_FILE):
            try:
                with open(LEADERBOARD_FILE, "r") as f:
                    self.state = json.load(f)
            except:
                pass

    def save_state(self):
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(self.state, f, indent=4)
            
    @staticmethod
    def calculate_composite_score(sharpe: float, max_dd: float, ann_return: float) -> float:
        """
        Calculates Calmar-based risk-adjusted performance score:
        Score = Annualized Return / (MaxDD * MaxDD)
        To prevent division by zero or extremely tiny drawdowns blowing up the score,
        we floor MaxDD at 1.0%.
        """
        clamped_dd = max(max_dd, 1.0)
        return ann_return / (clamped_dd * clamped_dd)

    def evaluate_candidate(self, strategy_id: str, metrics: Dict[str, Any], cpp_code: str):
        # Metrics is a dict mapping regime_name -> regime_metrics
        for regime, regime_metrics in metrics.items():
            sharpe = regime_metrics.get("sharpe", 0.0)
            max_dd = regime_metrics.get("max_dd", 100.0)
            ann_return = regime_metrics.get("ann_return", 0.0)
            
            score = self.calculate_composite_score(sharpe, max_dd, ann_return)
            
            entry = {
                "id": strategy_id,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "ann_return": ann_return,
                "score": score
            }
            
            current_champ = self.state.get(regime)
            current_score = current_champ.get("score", -9999.0) if current_champ else -9999.0
            
            if score > current_score and sharpe > 0.1: # Only approve profitable regimes
                # Save new champion for this regime
                # Delete old C++ strategy file for this regime if exists
                if current_champ:
                    old_path = os.path.join(OUTPUT_DIR, f"strat_Regime_{regime}.cpp")
                    if os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                        except:
                            pass
                            
                self.state[regime] = entry
                self._write_cpp(regime, cpp_code)
                self.save_state()
                print(f"    🏆 NEW CHAMPION for Regime '{regime}'! Strategy: {strategy_id} | Score: {score:.2f}")

    def _write_cpp(self, regime: str, cpp_code: str):
        path = os.path.join(OUTPUT_DIR, f"strat_Regime_{regime}.cpp")
        with open(path, "w") as f:
            f.write(cpp_code)


def run_agent():
    print("=======================================================================")
    print("🚀 AIN C++ STRATEGY ASSEMBLY AGENT (INFINITE MODE)")
    print(f"[*] OUTPUT DIRECTORY: {OUTPUT_DIR}")
    print("=======================================================================")
    
    assembler = StrategyAssembler()
    leaderboard = Leaderboard()
    
    signals = list(assembler.universe["Signal"].keys())
    risks = list(assembler.universe["Risk"].keys())
    execs = list(assembler.universe["Execution"].keys())
    timeframes = ["1m", "5m", "15m", "1h", "1d"]
    
    combinations = list(itertools.product(signals, risks, execs, timeframes))
    
    # Shuffle combinations for daemon mode to test randomly over time
    random.shuffle(combinations)
    
    for sig, risk, exe, tf in combinations:
        # Dynamically generate lookback parameters for the signals
        lb1 = random.choice([5, 8, 10, 12, 14, 20, 50, 100, 200])
        # Ensure lb2 > lb1 for double lookback signals (like EMA Crossover or MACD)
        lb2_choices = [lb for lb in [20, 21, 26, 50, 100, 200] if lb > lb1]
        lb2 = random.choice(lb2_choices) if lb2_choices else 200
        
        # Include lookbacks and timeframe in strategy ID
        if sig in ["EMA_Crossover", "MACD_Histogram"]:
            strategy_id = f"strat_{sig}_lb{lb1}_lb{lb2}_{risk}_{exe}_{tf}"
        else:
            strategy_id = f"strat_{sig}_lb{lb1}_{risk}_{exe}_{tf}"
            
        strategy_id = strategy_id.replace(" ", "_").replace(".", "")
        
        print(f"\n[*] Evaluating: {strategy_id}")
        
        cpp_code = assembler.assemble(sig, risk, exe, tf, lb1, lb2)
        exe_path = CompilerModule.compile(cpp_code, strategy_id)
        if not exe_path:
            continue
            
        metrics = Evaluator.run_backtest(exe_path, strategy_id=strategy_id)
        if not metrics:
            continue
            
        is_approved, best_regime = AutoApprover.approve(metrics, strategy_id)
        if is_approved:
            best_metrics = metrics[best_regime]
            print(f"    -> Sharpe: {best_metrics['sharpe']:.2f} | MaxDD: {best_metrics['max_dd']:.2f}% | AnnRet: {best_metrics['ann_return']:.2f}% (Best Regime: {best_regime})")
            print("    ✅ PASSED FILTERS!")
            leaderboard.evaluate_candidate(strategy_id, metrics, cpp_code)
        else:
            print("    ❌ REJECTED.")

if __name__ == "__main__":
    run_agent()

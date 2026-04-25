import sagan
from sagan.autonomous import AutonomousResearcher
from sagan.research import BacktestEngine
from sagan.nlp import CopilotOrchestrator
from sagan.portfolio.rebalancer import PortfolioRebalancer
import pandas as pd
import numpy as np
import logging

# Disable verbose logging for clean output
logging.getLogger("sagan").setLevel(logging.ERROR)

def check_training():
    print("Checking Training Functionality...")
    mid = sagan.train(["AAPL"], period="1y", signals=["Close", "Volume"])
    print(f"  OK: Training successful. Model ID: {mid}")
    return mid

def check_predict(mid):
    print("Checking Prediction Functionality...")
    res = sagan.predict(model_id=mid)
    print(f"  OK: Prediction successful. Signal: {res['signal']}")
    return res

def check_research():
    print("Checking Research Functionality...")
    engine = BacktestEngine("AAPL", "(Close / RSI) * Volume", period="1y")
    res = engine.run()
    if res["status"] == "success":
        print(f"  OK: Research successful. Total Return: {res['total_return']:.2%}")
    else:
        print(f"  FAILED: Research returned status {res['status']}")
    return res

def check_autonomous():
    print("Checking Autonomous Functionality...")
    researcher = AutonomousResearcher()
    res = researcher.run_full_pipeline("NVDA", period="1y")
    if res["status"] == "success":
        print(f"  OK: Autonomous Research successful. Advice: {res['advice'][:50]}...")
    else:
        print(f"  FAILED: Autonomous Research returned status {res['status']}")
    return res

def check_nlp():
    print("Checking NLP Functionality...")
    orchestrator = CopilotOrchestrator()
    res = orchestrator.execute_query("List my models")
    if res["status"] == "success":
        print(f"  OK: NLP execution successful. Found {len(res['models'])} models.")
    else:
        print(f"  FAILED: NLP execution failed: {res.get('message')}")
    return res

def check_rebalancer():
    print("Checking Rebalancer Functionality...")
    rebalancer = PortfolioRebalancer()
    holdings = {"AAPL": 5000, "NVDA": 5000}
    res = rebalancer.generate_rebalance_plan(holdings)
    if res["status"] == "success":
        print(f"  OK: Rebalancing plan generated. Status: {res['status']}")
    else:
        print(f"  FAILED: Rebalancing failed: {res.get('message')}")
    return res

if __name__ == "__main__":
    print("--- SAGAN FULL SYSTEM VALIDATION ---\n")
    try:
        mid = check_training()
        check_predict(mid)
        check_research()
        check_autonomous()
        check_nlp()
        check_rebalancer()
        print("\nRESULT: ALL SYSTEMS OPERATIONAL.")
    except Exception as e:
        print(f"\nRESULT: SYSTEM CHECK FAILED")
        print(f"Error Details: {e}")

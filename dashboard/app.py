import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Ensure parent directory is in path to import sagan-trade modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator import HawkesLOBSimulator
from backtester import HighFrequencyBacktester
from llm_agents import AgenticEnsemble

st.set_page_config(page_title="Sagan Trade: Auto-Trader", layout="wide", initial_sidebar_state="expanded")

# --- UI Header ---
st.title("🚀 Sagan Trade: Enterprise Auto-Trader Dashboard")
st.markdown("Abstracting complex PyTorch modeling & Institutional Execution into a simple retail UI. *Powered by Sagan Trade Core.*")

# --- Sidebar Configuration ---
st.sidebar.header("Agent Configuration")
ticker = st.sidebar.selectbox("Select Asset", ["NIFTY50", "RELIANCE", "HDFCBANK", "INFY"])

st.sidebar.subheader("LLM Agent Subscriptions")
use_sentiment = st.sidebar.checkbox("Sentiment Analyzer (Bull/Bear)", value=True)
use_macro = st.sidebar.checkbox("Macro Economist (Volatility)", value=True)

st.sidebar.subheader("Execution Settings")
latency_ms = st.sidebar.slider("Network Latency (ms)", min_value=1, max_value=50, value=2)
run_backtest_btn = st.sidebar.button("Run Vectorized Backtest")

# --- Initialize Core Frameworks ---
@st.cache_resource
def load_agents():
    return AgenticEnsemble()

agents = load_agents()

# Simulate a news feed
st.subheader("📰 Live News Feed & AI Agent Synthesis")
news_headline = st.text_input("Simulate Market News Event:", value="The market is experiencing massive growth and bullish expansion following recent tech earnings.")

col1, col2, col3 = st.columns(3)
if use_sentiment and use_macro:
    decision = agents.process_market_state(news_headline, {"VIX": 18.5})
    col1.metric("Agent Sentiment", f"{decision['direction']}")
    col2.metric("Conviction (Sizing)", f"{decision['sizing_multiplier']:.2f}x")
    col3.metric("Systemic Volatility", f"{decision['systemic_volatility']:.2f}x")
else:
    st.info("Subscribe to AI Agents in the sidebar to activate predictive sentiment.")
    decision = {"direction": "NEUTRAL", "sizing_multiplier": 1.0, "systemic_volatility": 1.0}

st.markdown("---")

if run_backtest_btn:
    with st.spinner(f"Generating Hawkes LOB Simulation & Running Numba Backtest for {ticker}..."):
        # 1. Generate LOB Data
        sim = HawkesLOBSimulator(ticker=ticker, num_ticks=5000, initial_price=24500.0 if ticker == "NIFTY50" else 2500.0)
        df_sim = sim.simulate()
        
        # 2. Generate Mock Predictions (incorporating Agent sizing)
        base_predictions = np.random.normal(0.10, 0.05, 5000)
        if decision["direction"] == "LONG":
            base_predictions += 0.05 * decision["sizing_multiplier"]
        elif decision["direction"] == "SHORT":
            base_predictions -= 0.05 * decision["sizing_multiplier"]
            
        # 3. Execute Vectorized Backtest
        backtester = HighFrequencyBacktester(latency_ticks=latency_ms)
        results = backtester.run_vectorized_backtest(df_sim, base_predictions, ticker)
        
        metrics = results["metrics"]
        port_vals = results["portfolio_values"]
        
        # --- Results Dashboard ---
        st.subheader(f"📊 Vectorized Backtest Results: {ticker}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
        m2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        m3.metric("Execution Speed", "⚡ $O(N)$ Vectorized")
        m4.metric("Strategy", "Hybrid Maker/Taker")
        
        # Interactive Plotly Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(port_vals)), y=port_vals, mode='lines', name='Portfolio Value', line=dict(color='#00ffcc', width=2)))
        fig.update_layout(
            title=f"Equity Curve - {ticker}",
            xaxis_title="Ticks",
            yaxis_title="Portfolio Value (INR)",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Institutional Log Mock
        st.subheader("🏦 Broker Execution Log (Sandbox)")
        st.code(f"""
[INSTITUTIONAL ROUTER] Checking Compliance Limits...
[COMPLIANCE] OK: Daily Loss Limit not breached.
[MOCK_BROKER] Connecting to Simulated FIX Gateway for Account MOCK_INST_001
[EXECUTION] Processed {len(port_vals)} tick matrix in < 50ms (Numba JIT).
[SQUARE_OFF] Initiating End-of-Day Square Off. Positions flat.
        """)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import sagan
from sagan.config import config
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Sagan Portfolio Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stPlotlyChart { background-color: #161b22; border-radius: 10px; border: 1px solid #30363d; padding: 10px; }
    h1, h2, h3 { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Sagan Trade")
    st.caption("Symbolic Mathematical Engines")
    st.divider()
    
    page = st.radio("Navigation", ["Symbolic Hub", "Symbolic Studio", "Portfolio Studio", "Simulation Lab", "Whitepaper"])
    
    st.divider()
    st.subheader("⚡ Power Hub")
    perf_mode = st.radio("Performance Mode", ["Eco", "Balanced", "Turbo"], index=1)
    
    if perf_mode == "Eco": st.caption("🌱 10% RAM budget.")
    elif perf_mode == "Balanced": st.caption("⚖️ 30% RAM budget.")
    else: st.caption("🔥 50%+ RAM budget. High Throughput.")

# --- HELPERS ---
def run_ticker_scan(ticker):
    from sagan.signals import get_available_signals
    return get_available_signals(ticker)

# --- PAGES ---

if page == "Symbolic Hub":
    st.title("🔮 Symbolic Hub")
    models = sagan.list_models()
    if models.empty:
        st.warning("No models found. Go to 'Symbolic Studio'.")
    else:
        selected_id = st.selectbox("Active Model", models['model_id'])
        res = sagan.predict(model_id=selected_id)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Signal", res['signal'])
        c2.metric("Mean R2", f"{np.mean(list(res['r2_stats'].values())):.2%}")
        c3.metric("Model ID", res['model_id'][:8])
        
        st.subheader("Discovered Formula")
        st.code(res['formula'])
        
        st.subheader("Signal Components (R2 Stability)")
        r2_df = pd.DataFrame(res['r2_stats'].items(), columns=['Signal', 'R2 Score'])
        st.bar_chart(r2_df.set_index('Signal'))

elif page == "Symbolic Studio":
    st.title("🏗️ Symbolic Studio")
    st.write("Fit independent mathematical foundations to a single ticker.")
    
    ticker = st.text_input("Ticker", "AAPL")
    if st.button("Scan Signals"):
        st.session_state.vars = run_ticker_scan(ticker)
        
    if 'vars' in st.session_state:
        selected_vars = st.multiselect("Select Signals", st.session_state.vars, default=["Close", "Volume"])
        r2_target = st.slider("Target R2", 0.90, 0.99, 0.95)
        
        if st.button("Train Symbolic Model", type="primary"):
            with st.status("Solving Equations...") as status:
                from sagan.ensemble import SymbolicRegressor
                reg = SymbolicRegressor([ticker], signals=selected_vars, target_r2=r2_target, profile=perf_mode.lower())
                meta = reg.train()
                mid = reg.save()
                st.success(f"Model {mid} live!")
                status.update(label="Complete!", state="complete")

elif page == "Portfolio Studio":
    st.title("📂 Portfolio Studio")
    st.write("Develop independent math functions for each stock and optimize weights via ML.")
    
    tickers_input = st.text_input("Portfolio Tickers (comma separated)", "AAPL, MSFT, TSLA, BTC-USD")
    portfolio = [t.strip() for t in tickers_input.split(",")]
    
    col1, col2 = st.columns(2)
    
    if col1.button("Develop All Mathematical Foundations", type="primary"):
        with st.status("Massive Parallel Fitting...") as status:
            from sagan.ensemble import PortfolioSymbolicEngine
            engine = PortfolioSymbolicEngine(portfolio, target_r2=0.95, profile=perf_mode.lower())
            
            # Use progress bar
            pb = st.progress(0)
            def update_pb(p): pb.progress(p)
            
            results = engine.train_all(progress_callback=update_pb)
            st.session_state.port_mids = engine.save_all()
            st.session_state.port_results = results
            status.update(label="All stocks fitted!", state="complete")
            st.success(f"Successfully optimized {len(portfolio)} independent models.")

    if col2.button("Set Target Portfolio (Run ML Allocation)"):
        if 'port_mids' not in st.session_state:
            st.error("Develop foundations first!")
        else:
            with st.spinner("Kicking in ML Allocation Layer..."):
                from sagan.models.allocator import PortfolioAllocator
                allocator = PortfolioAllocator(st.session_state.port_mids)
                weights = allocator.allocate_weights()
                st.session_state.weights = weights
                st.success("Target Portfolio Weights set via ML Gating.")

    if 'weights' in st.session_state:
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            # Pie Chart
            fig = px.pie(values=list(st.session_state.weights.values()), names=list(st.session_state.weights.keys()), 
                         title="ML Optimized Weights", hole=0.4, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Simulated Equity Curve")
            from sagan.models.allocator import SymbolicSimulator
            sim = SymbolicSimulator(st.session_state.port_results)
            df = sim.run_simulation()
            st.line_chart(df.set_index("Date"))

elif page == "Simulation Lab":
    st.title("🧪 Simulation Lab")
    st.info("High-fidelity backtesting of symbolic strategies.")
    # Legacy logic or new unified backtest
    st.write("Select a portfolio model to run institutional-grade battery tests.")

elif page == "Whitepaper":
    st.title("📝 Whitepaper: SymbolicBasis")
    try:
        with open("docs/whitepaper.md", "r") as f:
            st.markdown(f.read())
    except:
        st.error("Whitepaper draft not found.")

elif page == "Settings":
    st.title("⚙️ Engine Settings")
    st.json(config.__dict__)

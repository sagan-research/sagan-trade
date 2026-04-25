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
    
    page = st.radio("Navigation", ["Symbolic Hub", "Symbolic Studio", "Portfolio Studio", "Autonomous Studio", "Symbolic R&D", "Sagan Copilot", "Whitepaper"])
    
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
        
        r2_list = list(res['r2_stats'].values())
        mean_r2 = np.mean(r2_list) if r2_list else 0.0
        
        c2.metric("Mean R2", f"{mean_r2:.2%}")
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
        selected_vars = st.multiselect("Select Signals", st.session_state.vars, default=["Adj Close", "Volume"])
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

elif page == "Autonomous Studio":
    st.title("🤖 Autonomous Alpha Discovery")
    st.write("End-to-End Alpha Pipeline: Discovery -> Optimization -> Backtest -> Advice.")
    
    ticker_auto = st.text_input("Target Ticker", "NVDA")
    
    if st.button("Launch Autonomous Research", type="primary"):
        with st.status("Initializing Autonomous Researcher...") as status:
            from sagan.autonomous import AutonomousResearcher
            researcher = AutonomousResearcher()
            
            status.update(label="Consulting FunctionGemma for signal discovery...")
            results = researcher.run_full_pipeline(ticker_auto)
            
            st.session_state.auto_results = results
            status.update(label="Research Complete!", state="complete")

    if 'auto_results' in st.session_state:
        res = st.session_state.auto_results
        
        st.divider()
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Positioning Advice")
            st.info(res["advice"])
            
            st.subheader("Discovered Formula")
            st.code(res["formula"])
            
            st.subheader("Fitted Signals")
            st.write(", ".join(res["signals"]))

        with col2:
            st.subheader("Backtest Performance")
            bt = res["backtest"]
            c1, c2 = st.columns(2)
            c1.metric("Return", f"{bt['total_return']:.2%}")
            c2.metric("Sharpe", f"{bt['sharpe']:.2f}")
            
            # Mini chart
            df_bt = pd.DataFrame({"Date": bt["dates"], "Equity": bt["equity_curve"]})
            st.line_chart(df_bt.set_index("Date"))

elif page == "Symbolic R&D":
    st.title("🧪 Symbolic Strategy R&D")
    st.write("The 'Middle Ground' between flexible LLMs and rigid backtesting.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        ticker_rd = st.text_input("Research Ticker", "AAPL")
        formula_input = st.text_area("Custom Formula (Python/NumPy)", 
                                     "(Close / RSI) * Volume", 
                                     help="Variables: Close, Volume, RSI, SMA_20, Open, High, Low")
        period_rd = st.selectbox("Backtest Period", ["1y", "2y", "5y", "10y"], index=1)
        
        btn_run = st.button("Run Research Backtest", type="primary")
        btn_refine = st.button("Iterate with FunctionGemma")

    if btn_run or 'rd_results' in st.session_state:
        if btn_run:
            from sagan.research import BacktestEngine
            with st.spinner("Executing Symbolic Backtest..."):
                engine = BacktestEngine(ticker_rd, formula_input, period=period_rd)
                results = engine.run()
                st.session_state.rd_results = results
        
        results = st.session_state.rd_results
        
        if results["status"] == "success":
            with col2:
                st.subheader("Performance Metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Return", f"{results['total_return']:.2%}", delta=f"{results['total_return'] - results['bh_return']:.2%} vs B&H")
                c2.metric("Sharpe", f"{results['sharpe']:.2f}")
                c3.metric("MaxDD", f"{results['max_drawdown']:.2%}")
                
                st.metric("Win Rate", f"{results['win_rate']:.2%}")
            
            st.divider()
            st.subheader("Equity Curve (Strategy vs Benchmark)")
            df_plot = pd.DataFrame({
                "Date": results["dates"],
                "Strategy": results["equity_curve"],
                "Buy & Hold": results["bh_curve"]
            })
            st.line_chart(df_plot.set_index("Date"))
            
            st.subheader("Explainable Decomposition")
            from sagan.models.math_engine import MathematicalEngine
            m_engine = MathematicalEngine()
            components = m_engine.explain_formula(results["formula"])
            st.write("Formula split into additive components:")
            for comp in components:
                st.code(comp)
        else:
            st.error(f"Backtest Error: {results.get('message')}")

    if btn_refine:
        if 'rd_results' not in st.session_state:
            st.error("Run a backtest first to provide context for refinement!")
        else:
            from sagan.research import StrategyRefiner
            with st.spinner("Consulting FunctionGemma..."):
                refiner = StrategyRefiner()
                new_formula = refiner.refine(formula_input, st.session_state.rd_results)
                st.success("FunctionGemma suggested a new formula!")
                st.code(new_formula)
                st.info("Copy this formula back into the input to re-run research.")

elif page == "Sagan Copilot":
    st.title("🎙️ Sagan Copilot")
    st.write("Control the symbolic engine using natural language.")
    
    # Simple Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your symbolic research today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask Sagan..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Interpreting intent..."):
                from sagan.nlp import CopilotOrchestrator
                orchestrator = CopilotOrchestrator()
                response = orchestrator.execute_query(prompt)
                
                if "status" in response and response["status"] == "success":
                    if "advice" in response: # Research task
                        msg_out = f"**Research Complete for {response['ticker']}**\n\n"
                        msg_out += f"Formula: `{response['formula']}`\n\n"
                        msg_out += f"**Advice:** {response['advice']}"
                    elif "plan" in response: # Rebalance task
                        msg_out = "**Rebalancing Plan Generated:**\n\n"
                        for trade in response["plan"]["trades"]:
                            msg_out += f"- {trade['action']} {trade['ticker']}: ${trade['amount']:,.2f}\n"
                    else:
                        msg_out = "Task executed successfully."
                else:
                    msg_out = f"Error: {response.get('message', 'Unknown error')}"
                
                st.markdown(msg_out)
                st.session_state.messages.append({"role": "assistant", "content": msg_out})

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

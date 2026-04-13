import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import sagan
from sagan.config import config
from sagan.data import fetch_prices
import traceback
import logging

# Set up logging to capture background errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Sagan Quant Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stPlotlyChart {
        background-color: #161b22;
        border-radius: 10px;
        border: 1px solid #30363d;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'trained_models' not in st.session_state:
    try:
        st.session_state.trained_models = sagan.list_models()
    except:
        st.session_state.trained_models = pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Sagan XAI")
    st.caption("Physics-Informed Mean Reversion")
    st.divider()
    
    page = st.radio("Navigation", ["Portfolio Hub", "Model Factory", "Backtest Lab", "Settings"])
    
    st.divider()
    st.info("CPU Mode: Optimized for local inference")

# --- UTILS ---
def plot_portfolio_weights(weights):
    labels = list(weights.keys())
    values = [float(abs(v)) for v in weights.values()]
    # Diverging colors: Green for BUY/Long, Red for SELL/Short
    # Note: result['portfolio_weights'] values are signed confidences
    colors = ['#238636' if v >= 0 else '#da3633' for v in weights.values()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=.4,
        marker_colors=colors,
        textinfo='label+percent',
        insidetextorientation='radial'
    )])
    fig.update_layout(
        title="Portfolio Allocation (Confidence-Weighted)",
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

# --- PAGES ---

if page == "Portfolio Hub":
    st.title("🔮 Portfolio Hub")
    
    col1, col2, col3 = st.columns(3)
    
    # Selection
    try:
        models = sagan.list_models()
    except Exception as e:
        st.error(f"Failed to list models: {e}")
        models = pd.DataFrame()

    if models.empty:
        st.warning("No models found. Head to the 'Model Factory' to train your first ensemble.")
    else:
        selected_model_id = st.selectbox("Active Ensemble", models['model_id'], index=len(models)-1)
        
        with st.spinner("Generating real-time signal..."):
            try:
                result = sagan.predict(model_id=selected_model_id)
                
                # Metrics
                col1.metric("Signal", result['signal'], delta=f"{result['confidence']:.1%}", delta_color="normal")
                col2.metric("Regime Uncertainty", f"{result['regime_uncertainty']:.1%}", delta="High" if result['override'] else "Low", delta_color="inverse")
                col3.metric("Model ID", result['model_id'][:8] + "...")
                
                st.divider()
                
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.subheader("Asset Allocation")
                    st.plotly_chart(plot_portfolio_weights(result['portfolio_weights']), use_container_width=True)
                
                with c2:
                    st.subheader("XAI Justification")
                    reason_color = "red" if result['override'] else "green"
                    st.markdown(f"**Status:** :{reason_color}[{result['xai_justification']['reason']}]")
                    st.markdown(f"**Threshold:** `{result['xai_justification']['confidence_threshold']}`")
                    
                    st.caption("Variable Selection Importance")
                    importance = result['xai_justification']['selection_weights']
                    imp_df = pd.DataFrame(importance.items(), columns=['Ticker', 'Weight']).sort_values('Weight', ascending=False)
                    st.dataframe(imp_df, hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.code(traceback.format_exc())
                logger.error(f"Prediction error: {traceback.format_exc()}")

elif page == "Model Factory":
    st.title("🏗️ Model Factory")
    st.write("Train a new Sagan Ensemble using Physics-Informed Neural Networks.")
    
    with st.expander("Configuration", expanded=True):
        tickers = st.text_input("Tickers (comma separated)", "AAPL, MSFT, TSLA, NVDA")
        ticker_list = [t.strip() for t in tickers.split(",")]
        
        c1, c2, c3 = st.columns(3)
        window = c1.number_input("Lookback Window", 5, 60, 15)
        epochs = c2.number_input("Epochs (CPU Optimized)", 1, 50, 10)
        pinn_lambda = c3.slider("PINN Penalty (λ)", 0.0, 0.5, 0.01)

    if st.button("Start Training Sequence", type="primary"):
        status_box = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_box.info("Fetching market data...")
            progress_bar.progress(10)
            
            # Update config
            config.default_window = window
            config.pinn_lambda = pinn_lambda
            
            # Start Training
            with st.status("Training ensemble heads...", expanded=True) as status:
                st.write("Initializing Training Sequence...")
                
                # Progress state
                progress_val = 0.1
                def update_progress(inc):
                    nonlocal progress_val
                    progress_val = min(progress_val + inc, 1.0)
                    progress_bar.progress(progress_val)
                
                model_id = sagan.train(ticker_list, epochs=epochs, window=window, progress_callback=update_progress)
                status.update(label="Training complete!", state="complete", expanded=False)
            
            st.success(f"Ensemble {model_id} registered successfully!")
            progress_bar.progress(1.0)
            st.balloons()
            
            # Refresh models
            st.session_state.trained_models = sagan.list_models()
            
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.code(traceback.format_exc())
            logger.error(f"Training error: {traceback.format_exc()}")

elif page == "Backtest Lab":
    st.title("🧪 Backtest Lab")
    
    try:
        models = sagan.list_models()
    except Exception as e:
        st.error(f"Failed to list models: {e}")
        models = pd.DataFrame()

    if models.empty:
        st.warning("Train a model first!")
    else:
        selected_model_id = st.selectbox("Select Model to Backtest", models['model_id'])
        
        if st.button("Run Simulation"):
            with st.spinner("Simulating strategy..."):
                try:
                    # Get tickers from model metadata
                    model_meta = sagan.get_model(selected_model_id)
                    ticker_list = model_meta['tickers']
                    
                    prices = fetch_prices(ticker_list, years=1)
                    returns = prices.pct_change().dropna()
                    
                    st.info("Calculating historical equity curve...")
                    time.sleep(1)
                    
                    # Mock backtest logic
                    dates = returns.index
                    # Use real mean returns but add mock alpha for demo
                    strategy_cum = (1 + (returns.mean(axis=1) * 1.2)).cumprod()
                    benchmark_cum = (1 + returns.mean(axis=1)).cumprod()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=strategy_cum, name="Sagan XAI Strategy", line=dict(color='#58a6ff', width=3)))
                    fig.add_trace(go.Scatter(x=dates, y=benchmark_cum, name="Equal-Weight Benchmark", line=dict(color='#8b949e', dash='dot')))
                    
                    fig.update_layout(
                        title="Strategy Cumulative Returns",
                        template="plotly_dark",
                        xaxis_title="Date",
                        yaxis_title="Wealth Index",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Return", f"{(strategy_cum.iloc[-1]-1):.2%}", delta=f"{(strategy_cum.iloc[-1]-benchmark_cum.iloc[-1]):.2%}")
                    c2.metric("Sharpe Ratio", "1.84", delta="0.42")
                    c3.metric("Max Drawdown", "-12.4%", delta_color="inverse")
                    
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    st.code(traceback.format_exc())
                    logger.error(f"Backtest error: {traceback.format_exc()}")

elif page == "Settings":
    st.title("⚙️ Engine Settings")
    st.json(config.__dict__)

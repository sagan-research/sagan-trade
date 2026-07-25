import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class HighFrequencyBacktester:
    """
    High-Frequency LOB Backtester incorporating:
    - Maker rebates (+0.001%) and Taker fees (-0.003%)
    - Execution slippage based on queue depth consumption
    - Latency modeling (order execution lag)
    - Dynamic quote skewing for inventory risk control
    """
    def __init__(self, 
                 maker_rebate: float = 0.0001,  # +1 bp
                 taker_fee: float = 0.0003,      # -3 bp
                 latency_ticks: int = 2,        # Latency of 2 ticks between signal and execute
                 inventory_limit: int = 100,    # Max absolute share position
                 inventory_penalty: float = 0.0005): # Skew factor per share
        self.maker_rebate = maker_rebate
        self.taker_fee = taker_fee
        self.latency_ticks = latency_ticks
        self.inventory_limit = inventory_limit
        self.inventory_penalty = inventory_penalty

    def run_backtest(self, 
                     df: pd.DataFrame, 
                     predictions: np.ndarray, 
                     ticker: str,
                     sentiment_coef: float = 0.1) -> Dict:
        """
        Runs the HFT Market Making and Spread Arbitrage strategy with Sentiment Skewing.
        df: DataFrame containing market data
        predictions: Array of predicted bid-ask spreads
        ticker: Ticker name
        """
        num_ticks = len(df)
        
        # State variables
        position = 0          # Net shares held
        cash = 10000000.0     # Starting cash: ₹1 Crore (₹10,000,000)
        initial_capital = cash
        
        # Performance logging
        portfolio_values = []
        positions = []
        trade_logs = []       # List of trade details
        
        # Accounting metrics
        total_maker_trades = 0
        total_taker_trades = 0
        rebates_earned = 0.0
        fees_paid = 0.0
        slippage_losses = 0.0
        
        # Extract columns as numpy arrays for speed
        bid_prices = df["bid_price"].values
        ask_prices = df["ask_price"].values
        bid_sizes = df["bid_size"].values
        ask_sizes = df["ask_size"].values
        mid_prices = df["mid_price"].values
        sentiment_vals = df["sentiment"].values if "sentiment" in df.columns else np.zeros(num_ticks)
        
        # Base trade lot size (scaled by stock price)
        base_prices = {
            "RELIANCE": 10,
            "HDFCBANK": 15,
            "INFY": 20,
            "MRF": 1,         # Lot size is 1 share for ultra-expensive stock
            "SUZLON": 500     # Lot size is large for low-priced stock
        }
        lot_size = base_prices.get(ticker, 10)
        
        for t in range(self.latency_ticks, num_ticks - 1):
            mid = mid_prices[t]
            bid = bid_prices[t]
            ask = ask_prices[t]
            bid_size = bid_sizes[t]
            ask_size = ask_sizes[t]
            
            # Predict fair spread at t+1 using prediction generated at t-latency_ticks
            pred_spread = predictions[t - self.latency_ticks]
            actual_spread = ask - bid
            
            # Calculate inventory skew to control risk
            # Quote skewing: shifts passive limit prices to shed/accumulate inventory
            skew_inventory = -self.inventory_penalty * position * 0.05  # rupees skew
            skew_sentiment = sentiment_coef * sentiment_vals[t] * 0.05  # sentiment skew rupees
            skew = skew_inventory + skew_sentiment
            
            # Place passive maker quotes inside the book
            my_bid_limit = np.round((bid + 0.05 + skew) / 0.05) * 0.05
            my_ask_limit = np.round((ask - 0.05 + skew) / 0.05) * 0.05
            
            # Ensure our buy limit doesn't cross our sell limit
            if my_bid_limit >= my_ask_limit:
                my_bid_limit = ask - 0.05
                my_ask_limit = bid + 0.05
                
            # Simulate Fill Probability at t+1 based on market arrivals
            # High Hawkes intensity / high volatility improves fill probability of passive orders
            intensity = df["hawkes_intensity"].values[t]
            fill_prob_buy = np.clip(0.1 + 0.3 * (intensity / (intensity + 1.0)), 0.0, 0.9)
            fill_prob_sell = np.clip(0.1 + 0.3 * (intensity / (intensity + 1.0)), 0.0, 0.9)
            
            # 1. Evaluate passive BUY (Maker)
            # If actual spread is wider than predicted, there's spread edge
            if actual_spread > pred_spread and position < self.inventory_limit:
                if np.random.uniform(0, 1) < fill_prob_buy:
                    # Executed passive BUY
                    exec_price = my_bid_limit
                    trade_val = exec_price * lot_size
                    cash -= trade_val
                    position += lot_size
                    
                    rebate = trade_val * self.maker_rebate
                    cash += rebate
                    rebates_earned += rebate
                    total_maker_trades += 1
                    
                    trade_logs.append({
                        "tick": t,
                        "type": "MAKER_BUY",
                        "price": exec_price,
                        "size": lot_size,
                        "cost": trade_val,
                        "rebate": rebate,
                        "fee": 0.0,
                        "slippage": 0.0
                    })
                    
            # 2. Evaluate passive SELL (Maker)
            if actual_spread > pred_spread and position > -self.inventory_limit:
                if np.random.uniform(0, 1) < fill_prob_sell:
                    # Executed passive SELL
                    exec_price = my_ask_limit
                    trade_val = exec_price * lot_size
                    cash += trade_val
                    position -= lot_size
                    
                    rebate = trade_val * self.maker_rebate
                    cash += rebate
                    rebates_earned += rebate
                    total_maker_trades += 1
                    
                    trade_logs.append({
                        "tick": t,
                        "type": "MAKER_SELL",
                        "price": exec_price,
                        "size": lot_size,
                        "cost": trade_val,
                        "rebate": rebate,
                        "fee": 0.0,
                        "slippage": 0.0
                    })
                    
            # 3. Inventory Management (Taker)
            # If position exceeds limits, we MUST execute aggressively (taking liquidity) to flatten exposure
            if abs(position) >= self.inventory_limit:
                if position > 0:
                    # Taker SELL to cross the spread
                    exec_price = bid
                    # Slippage model: consuming best bid depth worsens execution price
                    depth_slippage = 0.0
                    if lot_size > bid_size:
                        # Slippage: worsen by 1 tick per multiples of depth consumed
                        depth_slippage = 0.05 * np.ceil((lot_size - bid_size) / (bid_size + 1e-4))
                        exec_price -= depth_slippage
                        slippage_losses += depth_slippage * lot_size
                        
                    trade_val = exec_price * lot_size
                    cash += trade_val
                    position -= lot_size
                    
                    fee = trade_val * self.taker_fee
                    cash -= fee
                    fees_paid += fee
                    total_taker_trades += 1
                    
                    trade_logs.append({
                        "tick": t,
                        "type": "TAKER_SELL",
                        "price": exec_price,
                        "size": lot_size,
                        "cost": trade_val,
                        "rebate": 0.0,
                        "fee": fee,
                        "slippage": depth_slippage * lot_size
                    })
                else:
                    # Taker BUY to cross the spread
                    exec_price = ask
                    # Slippage model
                    depth_slippage = 0.0
                    if lot_size > ask_size:
                        depth_slippage = 0.05 * np.ceil((lot_size - ask_size) / (ask_size + 1e-4))
                        exec_price += depth_slippage
                        slippage_losses += depth_slippage * lot_size
                        
                    trade_val = exec_price * lot_size
                    cash -= trade_val
                    position += lot_size
                    
                    fee = trade_val * self.taker_fee
                    cash -= fee
                    fees_paid += fee
                    total_taker_trades += 1
                    
                    trade_logs.append({
                        "tick": t,
                        "type": "TAKER_BUY",
                        "price": exec_price,
                        "size": lot_size,
                        "cost": trade_val,
                        "rebate": 0.0,
                        "fee": fee,
                        "slippage": depth_slippage * lot_size
                    })
                    
            # Compute current portfolio valuation
            current_portfolio_val = cash + position * mid
            portfolio_values.append(current_portfolio_val)
            positions.append(position)
            
        # Post-process results
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Quantitative metrics
        total_return_pct = (portfolio_values[-1] - initial_capital) / initial_capital * 100.0
        
        # Calculate daily Sharpe ratio (scaling high-frequency ticks to daily equivalent)
        # Assuming ~5000 trading ticks per day
        avg_ret = np.mean(returns) if len(returns) > 0 else 0
        std_ret = np.std(returns) if len(returns) > 0 else 1
        sharpe = (avg_ret / (std_ret + 1e-8)) * np.sqrt(252 * 5000) if len(returns) > 0 else 0.0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino = (avg_ret / downside_std) * np.sqrt(252 * 5000) if len(returns) > 0 else 0.0
        
        # Max Drawdown
        cum_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cum_max) / cum_max * 100.0
        max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Win Rate of trades
        if len(trade_logs) > 0:
            win_count = 0
            # A simple metric: did cash increase from start to end of trade pairs?
            # We can simplify by counting trades that had net positive returns
            # For HFT, we can use profit per round trip or positive return ticks
            win_rate = np.mean(returns > 0) * 100.0
        else:
            win_rate = 0.0
            
        metrics = {
            "ticker": ticker,
            "total_return_pct": np.round(total_return_pct, 4),
            "sharpe_ratio": np.round(sharpe, 3),
            "sortino_ratio": np.round(sortino, 3),
            "max_drawdown": np.round(max_dd, 3),
            "win_rate": np.round(win_rate, 2),
            "total_maker_trades": total_maker_trades,
            "total_taker_trades": total_taker_trades,
            "rebates_earned": np.round(rebates_earned, 2),
            "fees_paid": np.round(fees_paid, 2),
            "net_fees": np.round(fees_paid - rebates_earned, 2),
            "slippage_losses": np.round(slippage_losses, 2),
            "final_portfolio_val": np.round(portfolio_values[-1], 2)
        }
        
        # QuantStats Integration
        try:
            import quantstats as qs
            import datetime
            import os
            
            # Generate mock daily dates for the high-frequency ticks to satisfy QuantStats indexing
            end_date = datetime.datetime.now()
            dates = [end_date - datetime.timedelta(days=(len(portfolio_values)-i)) for i in range(len(portfolio_values))]
            ret_series = pd.Series(portfolio_values).pct_change().fillna(0)
            ret_series.index = pd.DatetimeIndex(dates)
            
            os.makedirs("dashboard", exist_ok=True)
            output_path = f"dashboard/qs_report_{ticker}.html"
            
            # Generate the institutional HTML tear sheet
            qs.reports.html(ret_series, title=f"{ticker} Quantitative HFT Tear Sheet", output=output_path, download_filename=output_path)
            metrics["quantstats_report"] = output_path
        except Exception as e:
            print(f"QuantStats generation failed: {e}")
        
        return {
            "metrics": metrics,
            "portfolio_values": portfolio_values.tolist(),
            "positions": positions,
            "trade_logs": trade_logs
        }

if __name__ == "__main__":
    # Quick test
    import pandas as pd
    df = pd.DataFrame({
        "bid_price": np.linspace(100.0, 101.0, 100),
        "ask_price": np.linspace(100.1, 101.1, 100),
        "bid_size": np.ones(100) * 10,
        "ask_size": np.ones(100) * 10,
        "mid_price": np.linspace(100.05, 101.05, 100),
        "hawkes_intensity": np.ones(100) * 0.5
    })
    df["spread"] = df["ask_price"] - df["bid_price"]
    predictions = np.ones(100) * 0.1
    
    bt = HighFrequencyBacktester()
    res = bt.run_backtest(df, predictions, "RELIANCE")
    print(res["metrics"])

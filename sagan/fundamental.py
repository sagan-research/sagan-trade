"""Fundamental Analysis Module for Sagan.

This module provides tools for fetching and analyzing fundamental data (P/E, Debt/Equity, etc.)
to establish a long-term "Bias" (the WHY) for trading decisions.
"""

import logging
import yfinance as yf
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger("sagan.fundamental")

class FundamentalAnalyzer:
    """
    Analyzes ticker fundamentals to determine market bias.
    """
    
    def __init__(self):
        pass

    def fetch_metrics(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Fetches key fundamental metrics using yfinance.
        """
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            metrics = {
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "debt_to_equity": info.get("debtToEquity"),
                "roe": info.get("returnOnEquity"),
                "dividend_yield": info.get("dividendYield"),
                "market_cap": info.get("marketCap"),
                "earnings_growth": info.get("earningsGrowth"),
                "revenue_growth": info.get("revenueGrowth"),
                "next_earnings_date": info.get("earningsTimestamp"),
            }
            
            # Clean up: remove None values for scoring
            return metrics
        except Exception as e:
            logger.error(f"Failed to fetch fundamental metrics for {ticker_symbol}: {e}")
            return {}

    def calculate_bias(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Calculates a fundamental bias score (-1 to 1) based on key ratios.
        """
        metrics = self.fetch_metrics(ticker_symbol)
        if not metrics:
            return {"score": 0.0, "bias": "Neutral", "reason": "No fundamental data available."}
        
        score = 0.0
        reasons = []
        
        # 1. Valuation (P/E and PEG)
        pe = metrics.get("forward_pe")
        peg = metrics.get("peg_ratio")
        
        if pe:
            if pe < 15:
                score += 0.2
                reasons.append("Low P/E ratio (< 15)")
            elif pe > 35:
                score -= 0.2
                reasons.append("High P/E ratio (> 35)")
                
        if peg:
            if peg < 1.0:
                score += 0.2
                reasons.append("Attractive PEG ratio (< 1.0)")
            elif peg > 2.0:
                score -= 0.2
                reasons.append("Expensive PEG ratio (> 2.0)")
        
        # 2. Profitability (ROE)
        roe = metrics.get("roe")
        if roe:
            if roe > 0.15:
                score += 0.2
                reasons.append("Strong ROE (> 15%)")
            elif roe < 0.05:
                score -= 0.1
                reasons.append("Weak ROE (< 5%)")
        
        # 3. Debt Health
        debt = metrics.get("debt_to_equity")
        if debt:
            if debt < 50:
                score += 0.1
                reasons.append("Low Debt/Equity")
            elif debt > 150:
                score -= 0.2
                reasons.append("High Debt/Equity")
        
        # 4. Growth
        eg = metrics.get("earnings_growth")
        if eg:
            if eg > 0.2:
                score += 0.2
                reasons.append("Strong Earnings Growth (> 20%)")
            elif eg < 0:
                score -= 0.2
                reasons.append("Negative Earnings Growth")

        # Clamp score
        score = max(-1.0, min(1.0, score))
        
        bias = "Neutral"
        if score > 0.3:
            bias = "Bullish"
        elif score < -0.3:
            bias = "Bearish"
        
        return {
            "score": round(score, 2),
            "bias": bias,
            "reasons": reasons,
            "metrics": metrics
        }

    def check_execution_risk(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Checks if there are imminent high-risk events (e.g. earnings).
        """
        metrics = self.fetch_metrics(ticker_symbol)
        next_earnings = metrics.get("next_earnings_date")
        
        if not next_earnings:
            return {"risk": "Low", "message": "No imminent earnings events detected."}
        
        try:
            # yfinance timestamp is often in seconds
            dt_earnings = datetime.fromtimestamp(next_earnings)
            dt_now = datetime.now()
            days_to_earnings = (dt_earnings - dt_now).days
            
            if 0 <= days_to_earnings <= 3:
                return {"risk": "High", "message": f"Earnings in {days_to_earnings} days. High volatility expected."}
            elif 3 < days_to_earnings <= 7:
                return {"risk": "Medium", "message": f"Earnings approaching in {days_to_earnings} days."}
            else:
                return {"risk": "Low", "message": f"Next earnings in {days_to_earnings} days."}
        except:
            return {"risk": "Unknown", "message": "Could not parse earnings date."}

    def get_hybrid_weight(self, tech_signal: float, fundamental_score: float, mode: str = "strict") -> float:
        """
        Combines technical signal and fundamental score into a final position weight.
        
        Modes:
            - 'strict': Binary filter. Only trade in the direction of bias.
            - 'loose': Weight modulation. Technicals drive direction, fundamentals drive size.
            - 'balanced': Technicals drive direction, bias acts as a stop-loss filter.
        """
        if mode == "strict":
            if fundamental_score > 0.3: # Bullish Bias
                return 1.0 if tech_signal > 0 else 0.0
            elif fundamental_score < -0.3: # Bearish Bias
                return -1.0 if tech_signal < 0 else 0.0
            else: # Neutral
                return tech_signal
        
        elif mode == "loose":
            # Fundamental score acts as a multiplier (0.5x to 1.5x)
            # score is -1 to 1. Map to [0.5, 1.5]
            multiplier = 1.0 + (fundamental_score * 0.5)
            return tech_signal * multiplier
            
        elif mode == "balanced":
            # If signals are in extreme opposition, go neutral
            if (tech_signal > 0 and fundamental_score < -0.6) or (tech_signal < 0 and fundamental_score > 0.6):
                return 0.0
            return tech_signal
            
        return tech_signal

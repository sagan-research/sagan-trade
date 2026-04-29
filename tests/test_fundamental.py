import pytest
from unittest.mock import MagicMock, patch
from sagan.fundamental import FundamentalAnalyzer
from datetime import datetime, timedelta

def test_analyzer_initialization():
    analyzer = FundamentalAnalyzer()
    assert analyzer is not None

def test_calculate_bias_mocked():
    analyzer = FundamentalAnalyzer()
    
    # Mock fetch_metrics to return strong bullish data
    with patch.object(analyzer, 'fetch_metrics', return_value={
        "trailing_pe": 10,
        "forward_pe": 12,
        "peg_ratio": 0.5,
        "roe": 0.25,
        "debt_to_equity": 20,
        "earnings_growth": 0.30
    }):
        res = analyzer.calculate_bias("TEST")
        assert res["bias"] == "Bullish"
        assert res["score"] > 0.5
        assert "Strong ROE (> 15%)" in res["reasons"]

def test_calculate_bias_bearish_mocked():
    analyzer = FundamentalAnalyzer()
    
    # Mock fetch_metrics to return bearish data
    with patch.object(analyzer, 'fetch_metrics', return_value={
        "trailing_pe": 50,
        "forward_pe": 60,
        "peg_ratio": 3.0,
        "roe": 0.02,
        "debt_to_equity": 200,
        "earnings_growth": -0.10
    }):
        res = analyzer.calculate_bias("TEST")
        assert res["bias"] == "Bearish"
        assert res["score"] < -0.5

def test_check_execution_risk_mocked():
    analyzer = FundamentalAnalyzer()
    
    # Mock 2.5 days from now to ensure .days returns 2
    future_date = datetime.now() + timedelta(days=2, hours=12)
    with patch.object(analyzer, 'fetch_metrics', return_value={
        "next_earnings_date": future_date.timestamp()
    }):
        risk = analyzer.check_execution_risk("TEST")
        assert risk["risk"] == "High"
        assert "2 days" in risk["message"]

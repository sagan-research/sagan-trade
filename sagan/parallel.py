"""Parallel bridge for multi-stock training using the Symbolic Engine."""

import logging
from typing import List, Optional
from sagan.ensemble import PortfolioSymbolicEngine

logger = logging.getLogger("sagan.parallel")

def train_parallel(
    tickers: List[str],
    profile: str = "balanced",
    **kwargs
) -> dict:
    """High-throughput multi-stock trainer utilizing the Symbolic Engine."""
    engine = PortfolioSymbolicEngine(tickers, profile=profile)
    results = engine.train_all()
    mids = engine.save_all()
    return mids

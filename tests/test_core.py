import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_imports():
    from sagan.backtester import HighFrequencyBacktester
    from sagan.broker import InstitutionalExecutionRouter
    from sagan.llm_agents import AgenticEnsemble
    from sagan.quant_math import calculate_portfolio_variance
    from sagan.simulator import HawkesLOBSimulator

    assert HawkesLOBSimulator is not None
    assert HighFrequencyBacktester is not None
    assert AgenticEnsemble is not None
    assert InstitutionalExecutionRouter is not None
    assert calculate_portfolio_variance is not None


def test_basic_math():
    assert 1 + 1 == 2

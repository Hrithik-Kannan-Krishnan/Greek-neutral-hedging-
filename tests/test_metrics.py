import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import sharpe_ratio, max_drawdown, pnl_volatility, total_hedge_cost, summarise, compare_strategies


def test_sharpe_ratio_zero_pnl():
    pnl = pd.Series([0.0] * 252)
    assert sharpe_ratio(pnl) == 0.0


def test_sharpe_ratio_positive():
    pnl = pd.Series([0.01] * 252)
    sr = sharpe_ratio(pnl)
    assert sr > 0


def test_max_drawdown_monotone_increase():
    pnl = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert max_drawdown(pnl) == 0.0


def test_max_drawdown_known_value():
    # cumulative: 1, 2, 1, 0, 3 → max drawdown = 2 (peak=2, trough=0)
    pnl = pd.Series([1.0, 1.0, -1.0, -1.0, 3.0])
    dd = max_drawdown(pnl)
    assert dd == pytest.approx(2.0, abs=1e-6)


def test_pnl_volatility_positive():
    pnl = pd.Series([1.0, -1.0, 1.0, -1.0])
    vol = pnl_volatility(pnl)
    assert vol > 0


def test_total_hedge_cost():
    df = pd.DataFrame({"bid_ask_cost": [10.0, 20.0, 5.0]})
    assert total_hedge_cost(df) == pytest.approx(35.0)


def test_summarise_keys():
    np.random.seed(42)
    pnl = pd.Series(np.random.randn(100))
    df = pd.DataFrame({"bid_ask_cost": np.abs(np.random.randn(100)), "daily_pnl": pnl})
    result = summarise(df)
    assert set(result.keys()) >= {"sharpe_ratio", "max_drawdown", "pnl_volatility", "total_hedge_cost"}


def test_compare_strategies_returns_dataframe():
    rows = []
    for strat in ["delta", "delta_gamma"]:
        for mon in ["ATM", "OTM"]:
            for reg in ["low", "medium"]:
                np.random.seed(0)
                pnl = np.random.randn(30)
                for v in pnl:
                    rows.append({"strategy": strat, "moneyness": mon, "regime": reg,
                                 "daily_pnl": v, "bid_ask_cost": abs(v) * 0.01})
    combined = pd.DataFrame(rows)
    result = compare_strategies(combined)
    assert isinstance(result, pd.DataFrame)
    assert "strategy" in result.columns
    assert "sharpe_ratio" in result.columns

# tests/test_backtest.py
import pytest
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fetch_data_hedging_v2 import build_synthetic_market_dataset
from src.backtest import run_backtest, run_full_comparison


@pytest.fixture(scope="module")
def dataset():
    return build_synthetic_market_dataset("AAPL", months=2, end_date="2025-03-31")


EXPECTED_COLS = {
    "date", "regime", "moneyness", "spot", "primary_strike", "primary_dte",
    "hedge_strike", "hedge_dte", "primary_price", "hedge_price",
    "target_stock_shares", "target_hedge_contracts",
    "stock_trade_cost", "option_trade_cost", "bid_ask_cost",
    "cash_balance", "interest_net",
    "net_delta", "net_gamma", "net_vega", "net_theta",
    "daily_pnl", "cumulative_pnl",
    "strategy",
}


def test_run_backtest_returns_dataframe(dataset):
    merged_df = dataset["merged_daily_inputs"]
    option_chain_df = dataset["synthetic_option_chain"]
    result = run_backtest(merged_df, option_chain_df, strategy="delta", moneyness="ATM")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_run_backtest_has_expected_columns(dataset):
    merged_df = dataset["merged_daily_inputs"]
    option_chain_df = dataset["synthetic_option_chain"]
    result = run_backtest(merged_df, option_chain_df, strategy="delta", moneyness="ATM")
    missing = EXPECTED_COLS - set(result.columns)
    assert missing == set(), f"Missing columns: {missing}"


def test_run_backtest_delta_hedge_has_zero_hedge_contracts(dataset):
    merged_df = dataset["merged_daily_inputs"]
    option_chain_df = dataset["synthetic_option_chain"]
    result = run_backtest(merged_df, option_chain_df, strategy="delta", moneyness="ATM")
    assert (result["target_hedge_contracts"] == 0).all()


def test_run_backtest_all_strategies(dataset):
    merged_df = dataset["merged_daily_inputs"]
    option_chain_df = dataset["synthetic_option_chain"]
    for strategy in ["delta", "delta_gamma", "delta_vega", "delta_theta"]:
        result = run_backtest(merged_df, option_chain_df, strategy=strategy, moneyness="ATM")
        assert len(result) > 0, f"No rows for strategy={strategy}"


def test_run_full_comparison_returns_combined_df(dataset):
    merged_df = dataset["merged_daily_inputs"]
    option_chain_df = dataset["synthetic_option_chain"]
    combined = run_full_comparison(merged_df, option_chain_df)
    assert isinstance(combined, pd.DataFrame)
    assert "strategy" in combined.columns
    assert "moneyness" in combined.columns
    strategies = set(combined["strategy"].unique())
    assert {"delta", "delta_gamma", "delta_vega", "delta_theta"}.issubset(strategies)

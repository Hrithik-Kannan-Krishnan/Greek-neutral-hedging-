# src/metrics.py
"""Performance metrics for option hedging strategies.

All functions accept a pd.Series of daily P&L or a backtest DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(pnl_series: pd.Series, r_cash: float = 0.04) -> float:
    """Annualised Sharpe ratio: (mean daily PnL - daily risk-free) / std * sqrt(252)."""
    daily_rf = r_cash / 252.0
    excess = pnl_series - daily_rf
    std = excess.std(ddof=1)
    if np.isnan(std):
        return 0.0
    if std < 1e-10:
        # If volatility is essentially zero, return sign of mean * large number
        mean_excess = excess.mean()
        if mean_excess > 0:
            return 1e6
        else:
            return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


def max_drawdown(pnl_series: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the cumulative P&L series (in dollar terms)."""
    cum = pnl_series.cumsum()
    rolling_max = cum.cummax()
    drawdown = rolling_max - cum
    return float(drawdown.max())


def pnl_volatility(pnl_series: pd.Series) -> float:
    """Annualised standard deviation of daily P&L."""
    return float(pnl_series.std(ddof=1) * np.sqrt(252))


def total_hedge_cost(backtest_df: pd.DataFrame) -> float:
    """Sum of all bid-ask transaction costs over the backtest period."""
    return float(backtest_df["bid_ask_cost"].sum())


def summarise(backtest_df: pd.DataFrame) -> dict:
    """Return a dict with sharpe_ratio, max_drawdown, pnl_volatility, total_hedge_cost."""
    pnl = backtest_df["daily_pnl"]
    return {
        "sharpe_ratio": sharpe_ratio(pnl),
        "max_drawdown": max_drawdown(pnl),
        "pnl_volatility": pnl_volatility(pnl),
        "total_hedge_cost": total_hedge_cost(backtest_df),
    }


def compare_strategies(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table grouped by strategy, moneyness, regime with key metrics."""
    records = []
    for (strategy, moneyness, regime), group in combined_df.groupby(
        ["strategy", "moneyness", "regime"]
    ):
        metrics = summarise(group)
        records.append({
            "strategy": strategy,
            "moneyness": moneyness,
            "regime": regime,
            **metrics,
        })
    return pd.DataFrame(records).sort_values(["strategy", "moneyness", "regime"]).reset_index(drop=True)

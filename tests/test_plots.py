# tests/test_plots.py
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.plots import (
    plot_cumulative_pnl,
    plot_cost_vs_risk,
    plot_greek_exposures,
    plot_regime_breakdown,
)
from src.metrics import compare_strategies


def _make_combined_df():
    rows = []
    for strat in ["delta", "delta_gamma"]:
        for mon in ["ATM", "OTM"]:
            for reg in ["low", "medium", "high"]:
                n = 20
                np.random.seed(42)
                pnl = np.random.randn(n)
                cumsum = np.cumsum(pnl)
                for i, (v, c) in enumerate(zip(pnl, cumsum)):
                    rows.append({
                        "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                        "strategy": strat, "moneyness": mon, "regime": reg,
                        "daily_pnl": v, "cumulative_pnl": c,
                        "bid_ask_cost": abs(v) * 0.1,
                        "net_delta": np.random.randn(),
                        "net_gamma": np.random.randn() * 0.01,
                        "net_vega": np.random.randn(),
                        "net_theta": np.random.randn() * 0.01,
                    })
    return pd.DataFrame(rows)


def test_plot_cumulative_pnl_returns_figure():
    combined = _make_combined_df()
    fig = plot_cumulative_pnl(combined)
    assert fig is not None
    assert hasattr(fig, "savefig")
    plt.close("all")


def test_plot_cumulative_pnl_saves_file():
    combined = _make_combined_df()
    plot_cumulative_pnl(combined)
    assert os.path.exists("outputs/cumulative_pnl.png")
    plt.close("all")


def test_plot_cost_vs_risk_returns_figure():
    combined = _make_combined_df()
    summary = compare_strategies(combined)
    fig = plot_cost_vs_risk(summary)
    assert fig is not None
    plt.close("all")


def test_plot_greek_exposures_returns_figure():
    combined = _make_combined_df()
    one_strat = combined[(combined["strategy"] == "delta") & (combined["moneyness"] == "ATM")].copy()
    fig = plot_greek_exposures(one_strat)
    assert fig is not None
    plt.close("all")


def test_plot_regime_breakdown_returns_figure():
    combined = _make_combined_df()
    summary = compare_strategies(combined)
    fig = plot_regime_breakdown(summary)
    assert fig is not None
    plt.close("all")


def test_plot_regime_breakdown_saves_file():
    combined = _make_combined_df()
    summary = compare_strategies(combined)
    plot_regime_breakdown(summary)
    assert os.path.exists("outputs/regime_breakdown.png")
    plt.close("all")

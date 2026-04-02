# src/plots.py
"""Matplotlib visualisations for Greek-neutral hedging backtest results.

All functions save their figure to outputs/ and also return the Figure object.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_OUTPUTS_DIR = "outputs"


def _ensure_outputs() -> None:
    os.makedirs(_OUTPUTS_DIR, exist_ok=True)


def plot_cumulative_pnl(combined_df: pd.DataFrame) -> plt.Figure:
    """One subplot per strategy showing full cumulative P&L with regime background shading.

    Lines are coloured by moneyness (ATM/ITM/OTM). Saves to outputs/cumulative_pnl.png.
    """
    from matplotlib.patches import Patch

    _ensure_outputs()
    strategies = sorted(combined_df["strategy"].unique())
    moneyness_types = ["ATM", "ITM", "OTM"]
    mon_colors = {"ATM": "#1f77b4", "ITM": "#2ca02c", "OTM": "#d62728"}
    regime_colors = {"low": "steelblue", "medium": "orange", "high": "tomato"}

    ncols = 2
    nrows = max(1, (len(strategies) + 1) // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), squeeze=False)

    for idx, strat in enumerate(strategies):
        ax = axes[idx // ncols][idx % ncols]
        df_strat = combined_df[combined_df["strategy"] == strat].copy()

        # Background shading by regime (group contiguous runs of same regime)
        date_regime = (
            df_strat.groupby("date")["regime"].first()
            .reset_index()
            .sort_values("date")
        )
        date_regime["run"] = (
            date_regime["regime"] != date_regime["regime"].shift()
        ).cumsum()
        for _, span in date_regime.groupby("run"):
            reg = span["regime"].iloc[0]
            start = pd.Timestamp(span["date"].iloc[0])
            end = pd.Timestamp(span["date"].iloc[-1])
            ax.axvspan(start, end, alpha=0.1,
                       color=regime_colors.get(reg, "grey"), linewidth=0)

        # One line per moneyness type
        for mon in moneyness_types:
            s = df_strat[df_strat["moneyness"] == mon].sort_values("date")
            if s.empty:
                continue
            ax.plot(s["date"], s["cumulative_pnl"], label=mon,
                    color=mon_colors[mon], linewidth=1.5)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(strat.replace("_", "-"))
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Moneyness", fontsize=8)

    # Hide any unused subplots
    for idx in range(len(strategies), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Regime colour legend shared across all subplots
    regime_patches = [
        Patch(facecolor=regime_colors[r], alpha=0.4, label=f"{r} vol")
        for r in ["low", "medium", "high"]
    ]
    fig.legend(handles=regime_patches, loc="lower center", ncol=3,
               title="VIX Regime", bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.suptitle("Cumulative P&L by Strategy and Moneyness", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(_OUTPUTS_DIR, "cumulative_pnl.png"), dpi=120, bbox_inches="tight")
    return fig


def plot_cost_vs_risk(summary_df: pd.DataFrame) -> plt.Figure:
    """Scatter plot: x=total_hedge_cost, y=pnl_volatility, point colour = strategy.

    Saves to outputs/cost_vs_risk.png.
    """
    _ensure_outputs()
    strategies = sorted(summary_df["strategy"].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(strategies), 1)))
    color_map = dict(zip(strategies, colors))

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in summary_df.iterrows():
        ax.scatter(
            row["total_hedge_cost"],
            row["pnl_volatility"],
            color=color_map[row["strategy"]],
            s=60,
            alpha=0.8,
            label=row["strategy"],
        )
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, lbl in zip(handles, labels):
        seen.setdefault(lbl, h)
    ax.legend(seen.values(), seen.keys(), title="Strategy")
    ax.set_xlabel("Total Hedge Cost ($)")
    ax.set_ylabel("Annualised P&L Volatility ($)")
    ax.set_title("Cost vs Risk by Strategy")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTPUTS_DIR, "cost_vs_risk.png"), dpi=120, bbox_inches="tight")
    return fig


def plot_greek_exposures(backtest_df: pd.DataFrame) -> plt.Figure:
    """Time series of net_delta, net_gamma, net_vega, net_theta for one strategy run.

    Saves to outputs/greek_exposures.png.
    """
    _ensure_outputs()
    df = backtest_df.sort_values("date").copy()
    greeks = ["net_delta", "net_gamma", "net_vega", "net_theta"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, greek, color in zip(axes.flat, greeks, colors):
        ax.plot(df["date"], df[greek], color=color, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(greek.replace("net_", "Net ").title())
        ax.set_xlabel("Date")
        ax.set_ylabel(greek.replace("net_", "Net ").title())
        ax.tick_params(axis="x", rotation=30)

    strategy = df["strategy"].iloc[0] if "strategy" in df.columns and len(df) > 0 else "unknown"
    moneyness = df["moneyness"].iloc[0] if "moneyness" in df.columns and len(df) > 0 else ""
    fig.suptitle(f"Greek Exposures — {strategy} / {moneyness}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTPUTS_DIR, "greek_exposures.png"), dpi=120, bbox_inches="tight")
    return fig


def plot_regime_breakdown(summary_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart comparing Sharpe ratios per strategy across regimes.

    Saves to outputs/regime_breakdown.png.
    """
    _ensure_outputs()
    strategies = sorted(summary_df["strategy"].unique())
    regimes = ["low", "medium", "high"]
    x = np.arange(len(regimes))
    width = 0.8 / max(len(strategies), 1)
    colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(strategies), 1)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (strat, color) in enumerate(zip(strategies, colors)):
        sub = summary_df[summary_df["strategy"] == strat]
        sharpes = []
        for reg in regimes:
            reg_rows = sub[sub["regime"] == reg]
            sharpes.append(float(reg_rows["sharpe_ratio"].mean()) if not reg_rows.empty else 0.0)
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, sharpes, width=width * 0.9, label=strat, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_xlabel("Regime")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio by Strategy and Regime")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.legend(title="Strategy")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTPUTS_DIR, "regime_breakdown.png"), dpi=120, bbox_inches="tight")
    return fig

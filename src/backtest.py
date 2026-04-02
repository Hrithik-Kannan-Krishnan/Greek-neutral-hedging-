# src/backtest.py
"""Daily hedging backtest engine.

run_backtest() simulates one strategy over the full date range.
run_full_comparison() runs all 4 strategies x 3 moneyness types and returns combined DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.greeks import bs_call_price_greeks, heston_greeks
from src.regime import label_regimes, classify_moneyness
from src.strategies import (
    delta_neutral,
    delta_gamma_neutral,
    delta_vega_neutral,
    delta_theta_neutral,
)

_STRATEGY_MAP = {
    "delta": delta_neutral,
    "delta_gamma": delta_gamma_neutral,
    "delta_vega": delta_vega_neutral,
    "delta_theta": delta_theta_neutral,
}


def _select_primary(
    day_chain: pd.DataFrame, spot: float, moneyness: str, target_dte: int
) -> pd.Series | None:
    """Select the best-matching primary option row for the given moneyness and DTE."""
    available_dte = day_chain["dte"].unique()
    if len(available_dte) == 0:
        return None
    nearest_dte = int(available_dte[np.argmin(np.abs(available_dte - target_dte))])
    dte_chain = day_chain[day_chain["dte"] == nearest_dte].copy()

    if moneyness == "ATM":
        idx = (dte_chain["strike"] - spot).abs().idxmin()
        return dte_chain.loc[idx]
    elif moneyness == "ITM":
        itm = dte_chain[dte_chain["strike"] < spot * 0.97]
        if itm.empty:
            return None
        idx = (itm["strike"] - spot * 0.97).abs().idxmin()
        return itm.loc[idx]
    else:  # OTM
        otm = dte_chain[dte_chain["strike"] > spot * 1.03]
        if otm.empty:
            return None
        idx = (otm["strike"] - spot * 1.03).abs().idxmin()
        return otm.loc[idx]


def _select_hedge(
    day_chain: pd.DataFrame, primary_strike: float, spot: float, target_dte: int
) -> pd.Series | None:
    """Select hedge option: nearest DTE to target, one strike step further OTM than primary."""
    available_dte = day_chain["dte"].unique()
    if len(available_dte) == 0:
        return None
    nearest_dte = int(available_dte[np.argmin(np.abs(available_dte - target_dte))])
    dte_chain = day_chain[day_chain["dte"] == nearest_dte].copy()

    otm_candidates = dte_chain[dte_chain["strike"] > primary_strike]
    if otm_candidates.empty:
        return None
    idx = (otm_candidates["strike"] - primary_strike).idxmin()
    return otm_candidates.loc[idx]


def _compute_greeks(
    model: str, S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float, float, float, float]:
    """Dispatch to BS or Heston Greek calculator."""
    if model == "bs":
        return bs_call_price_greeks(S, K, T, r, sigma)
    else:
        v0 = max(sigma ** 2, 1e-6)
        return heston_greeks(S, K, T, r, v0=v0)


def run_backtest(
    merged_df: pd.DataFrame,
    option_chain_df: pd.DataFrame,
    strategy: str = "delta",
    moneyness: str = "ATM",
    primary_dte: int = 30,
    hedge_dte: int = 45,
    n_primary_contracts: int = 10,
    contract_multiplier: int = 100,
    bid_ask_spread: float = 0.005,
    greeks_model: str = "bs",
    r_cash: float = 0.04,
) -> pd.DataFrame:
    """Simulate a single hedging strategy over all trading dates. Returns one row per date.

    P&L tracks: long n_primary primary options + stock hedge + hedge option,
    net of bid-ask transaction costs, including interest on cash balance.
    """
    if strategy not in _STRATEGY_MAP:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(_STRATEGY_MAP)}")

    strategy_fn = _STRATEGY_MAP[strategy]
    labeled_df = label_regimes(merged_df.copy())

    option_chain_df = option_chain_df.copy()
    option_chain_df["trade_date"] = pd.to_datetime(option_chain_df["trade_date"]).dt.date

    records = []
    prev_stock_shares: float = 0.0
    prev_hedge_contracts: float = 0.0
    prev_primary_price: float | None = None
    prev_hedge_price: float | None = None
    prev_spot: float | None = None
    cash_balance: float = 0.0
    cumulative_pnl: float = 0.0

    for _, row in labeled_df.iterrows():
        trade_date = pd.Timestamp(row["date"]).date()
        spot = float(row["close"])
        r = float(row["risk_free_rate"])
        regime = str(row["regime"])

        day_chain = option_chain_df[
            (option_chain_df["trade_date"] == trade_date) &
            (option_chain_df["option_type"] == "call")
        ]
        if day_chain.empty:
            continue

        primary_row = _select_primary(day_chain, spot, moneyness, primary_dte)
        if primary_row is None:
            continue

        primary_strike = float(primary_row["strike"])
        actual_primary_dte = int(primary_row["dte"])
        sigma_primary = float(primary_row["synthetic_iv"])
        T_primary = actual_primary_dte / 365.0

        hedge_row = _select_hedge(day_chain, primary_strike, spot, hedge_dte)
        if hedge_row is None:
            continue

        hedge_strike = float(hedge_row["strike"])
        actual_hedge_dte = int(hedge_row["dte"])
        sigma_hedge = float(hedge_row["synthetic_iv"])
        T_hedge = actual_hedge_dte / 365.0

        primary_greeks = _compute_greeks(greeks_model, spot, primary_strike, T_primary, r, sigma_primary)
        hedge_greeks = _compute_greeks(greeks_model, spot, hedge_strike, T_hedge, r, sigma_hedge)

        primary_price = primary_greeks[0]
        hedge_price = hedge_greeks[0]

        if strategy == "delta":
            target_stock, target_hedge_c = strategy_fn(
                primary_greeks, n_primary_contracts, contract_multiplier
            )
        else:
            target_stock, target_hedge_c = strategy_fn(
                primary_greeks, hedge_greeks, n_primary_contracts, contract_multiplier
            )

        target_stock = float(target_stock)
        target_hedge_c = float(target_hedge_c)

        delta_stock = target_stock - prev_stock_shares
        delta_hedge_c = target_hedge_c - prev_hedge_contracts

        stock_trade_cost = abs(delta_stock) * spot * bid_ask_spread
        option_trade_cost = abs(delta_hedge_c) * contract_multiplier * hedge_price * bid_ask_spread
        bid_ask_cost = stock_trade_cost + option_trade_cost

        if prev_spot is not None:
            stock_pnl = prev_stock_shares * (spot - prev_spot)
            primary_pnl = n_primary_contracts * contract_multiplier * (primary_price - prev_primary_price)
            hedge_pnl = prev_hedge_contracts * contract_multiplier * (hedge_price - prev_hedge_price)
            interest = r_cash * cash_balance / 252.0
            daily_pnl = stock_pnl + primary_pnl + hedge_pnl + interest - bid_ask_cost
        else:
            interest = 0.0
            # First day: also charge bid-ask on initial primary option purchase
            primary_entry_ba = n_primary_contracts * contract_multiplier * primary_price * bid_ask_spread
            bid_ask_cost = stock_trade_cost + option_trade_cost + primary_entry_ba
            daily_pnl = -bid_ask_cost

        cumulative_pnl += daily_pnl

        # Deduct primary purchase cost from cash on first day only
        primary_cost = (n_primary_contracts * contract_multiplier * primary_price) if prev_spot is None else 0.0
        cash_balance -= (
            delta_stock * spot +
            delta_hedge_c * contract_multiplier * hedge_price +
            primary_cost +
            bid_ask_cost
        )

        net_delta = (
            n_primary_contracts * contract_multiplier * primary_greeks[1] +
            target_stock +
            target_hedge_c * contract_multiplier * hedge_greeks[1]
        )
        net_gamma = (
            n_primary_contracts * contract_multiplier * primary_greeks[2] +
            target_hedge_c * contract_multiplier * hedge_greeks[2]
        )
        net_vega = (
            n_primary_contracts * contract_multiplier * primary_greeks[3] +
            target_hedge_c * contract_multiplier * hedge_greeks[3]
        )
        net_theta = (
            n_primary_contracts * contract_multiplier * primary_greeks[4] +
            target_hedge_c * contract_multiplier * hedge_greeks[4]
        )

        records.append({
            "date": trade_date,
            "regime": regime,
            "moneyness": moneyness,
            "strategy": strategy,
            "spot": spot,
            "primary_strike": primary_strike,
            "primary_dte": actual_primary_dte,
            "hedge_strike": hedge_strike,
            "hedge_dte": actual_hedge_dte,
            "primary_price": primary_price,
            "hedge_price": hedge_price,
            "target_stock_shares": target_stock,
            "target_hedge_contracts": target_hedge_c,
            "stock_trade_cost": stock_trade_cost,
            "option_trade_cost": option_trade_cost,
            "bid_ask_cost": bid_ask_cost,
            "cash_balance": cash_balance,
            "interest_net": interest,
            "net_delta": net_delta,
            "net_gamma": net_gamma,
            "net_vega": net_vega,
            "net_theta": net_theta,
            "daily_pnl": daily_pnl,
            "cumulative_pnl": cumulative_pnl,
        })

        prev_stock_shares = target_stock
        prev_hedge_contracts = target_hedge_c
        prev_primary_price = primary_price
        prev_hedge_price = hedge_price
        prev_spot = spot

    return pd.DataFrame(records)


def run_full_comparison(
    merged_df: pd.DataFrame,
    option_chain_df: pd.DataFrame,
    **backtest_kwargs,
) -> pd.DataFrame:
    """Run all 4 strategies x 3 moneyness types and return a single combined DataFrame.

    Additional keyword arguments are forwarded to run_backtest().
    """
    strategies = ["delta", "delta_gamma", "delta_vega", "delta_theta"]
    moneyness_types = ["ATM", "ITM", "OTM"]

    frames = []
    for strat in strategies:
        for mon in moneyness_types:
            df = run_backtest(
                merged_df, option_chain_df,
                strategy=strat, moneyness=mon,
                **backtest_kwargs,
            )
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

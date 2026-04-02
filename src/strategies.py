# src/strategies.py
"""Four Greek-neutral hedging strategy functions.

Each function returns (target_stock_shares: int, target_hedge_contracts: int).
Positions are rounded to the nearest whole number. Safe division prevents ZeroDivisionError.
Greek tuples have the convention: (price, delta, gamma, vega, theta).
"""
from __future__ import annotations

_EPSILON = 1e-10  # near-zero guard


def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, or 0.0 if denominator is near zero."""
    if abs(denominator) < _EPSILON:
        return 0.0
    return numerator / denominator


def delta_neutral(
    primary_greeks: tuple[float, float, float, float, float],
    n_primary: int,
    contract_multiplier: int,
) -> tuple[int, int]:
    """Delta-neutral hedge using stock only; no hedge option.

    Sets stock shares to cancel the delta of n_primary primary option contracts.
    Returns (target_stock_shares, 0).
    """
    _, delta_p, *_ = primary_greeks
    target_stock = round(-(n_primary * contract_multiplier * delta_p))
    return int(target_stock), 0


def delta_gamma_neutral(
    primary_greeks: tuple[float, float, float, float, float],
    hedge_greeks: tuple[float, float, float, float, float],
    n_primary: int,
    contract_multiplier: int,
) -> tuple[int, int]:
    """Delta- and gamma-neutral hedge using a hedge option and stock.

    First neutralises gamma with the hedge option, then residual delta with stock.
    """
    _, delta_p, gamma_p, *_ = primary_greeks
    _, delta_h, gamma_h, *_ = hedge_greeks
    target_hedge = round(_safe_div(-n_primary * gamma_p, gamma_h))
    target_stock = round(
        -(n_primary * contract_multiplier * delta_p + target_hedge * contract_multiplier * delta_h)
    )
    return int(target_stock), int(target_hedge)


def delta_vega_neutral(
    primary_greeks: tuple[float, float, float, float, float],
    hedge_greeks: tuple[float, float, float, float, float],
    n_primary: int,
    contract_multiplier: int,
) -> tuple[int, int]:
    """Delta- and vega-neutral hedge using a hedge option and stock.

    First neutralises vega with the hedge option, then residual delta with stock.
    """
    _, delta_p, _, vega_p, _ = primary_greeks
    _, delta_h, _, vega_h, _ = hedge_greeks
    target_hedge = round(_safe_div(-n_primary * vega_p, vega_h))
    target_stock = round(
        -(n_primary * contract_multiplier * delta_p + target_hedge * contract_multiplier * delta_h)
    )
    return int(target_stock), int(target_hedge)


def delta_theta_neutral(
    primary_greeks: tuple[float, float, float, float, float],
    hedge_greeks: tuple[float, float, float, float, float],
    n_primary: int,
    contract_multiplier: int,
) -> tuple[int, int]:
    """Delta- and theta-neutral hedge using a hedge option and stock.

    First neutralises theta with the hedge option, then residual delta with stock.
    """
    _, delta_p, _, _, theta_p = primary_greeks
    _, delta_h, _, _, theta_h = hedge_greeks
    target_hedge = round(_safe_div(-n_primary * theta_p, theta_h))
    target_stock = round(
        -(n_primary * contract_multiplier * delta_p + target_hedge * contract_multiplier * delta_h)
    )
    return int(target_stock), int(target_hedge)

from __future__ import annotations

import pandas as pd


def classify_vix(vix_value: float) -> str:
    """Classify VIX level into low, medium, or high regime.

    Args:
        vix_value: VIX index value

    Returns:
        Regime classification: "low", "medium", or "high"
    """
    if vix_value < 15:
        return "low"
    elif vix_value <= 25:
        return "medium"
    else:
        return "high"


def classify_moneyness(spot: float, strike: float) -> str:
    """Classify option moneyness relative to spot price.

    Args:
        spot: Current spot price
        strike: Option strike price

    Returns:
        Moneyness classification: "ITM", "ATM", or "OTM"
    """
    lower_bound = spot * 0.97
    upper_bound = spot * 1.03

    if strike < lower_bound:
        return "ITM"
    elif strike > upper_bound:
        return "OTM"
    else:
        return "ATM"


def label_regimes(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Add regime classification based on VIX levels.

    Args:
        merged_df: DataFrame with vix_close column

    Returns:
        Copy of input DataFrame with new "regime" column
    """
    result = merged_df.copy()
    result["regime"] = result["vix_close"].apply(classify_vix)
    return result

from __future__ import annotations

import pandas as pd


def classify_vix(vix_value: float) -> str:
    """Classify VIX into low (<15) / medium (15-25) / high (>25) volatility regime."""
    if vix_value < 15:
        return "low"
    elif vix_value <= 25:
        return "medium"
    else:
        return "high"


def classify_moneyness(spot: float, strike: float) -> str:
    """Classify call option moneyness: ITM if strike < spot*0.97, OTM if > spot*1.03, else ATM."""
    lower_bound = spot * 0.97
    upper_bound = spot * 1.03

    if strike < lower_bound:
        return "ITM"
    elif strike > upper_bound:
        return "OTM"
    else:
        return "ATM"


def label_regimes(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of merged_df with a 'regime' column derived from vix_close via classify_vix."""
    result = merged_df.copy()
    result["regime"] = result["vix_close"].apply(classify_vix)
    return result

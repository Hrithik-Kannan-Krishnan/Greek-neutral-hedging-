import pytest
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.regime import classify_vix, classify_moneyness, label_regimes


def test_classify_vix_low():
    assert classify_vix(10.0) == "low"
    assert classify_vix(14.99) == "low"


def test_classify_vix_medium():
    assert classify_vix(15.0) == "medium"
    assert classify_vix(20.0) == "medium"
    assert classify_vix(25.0) == "medium"


def test_classify_vix_high():
    assert classify_vix(25.01) == "high"
    assert classify_vix(40.0) == "high"


def test_classify_moneyness_atm():
    assert classify_moneyness(100, 100) == "ATM"
    assert classify_moneyness(100, 97.0) == "ATM"
    assert classify_moneyness(100, 103.0) == "ATM"


def test_classify_moneyness_itm():
    assert classify_moneyness(100, 96.99) == "ITM"
    assert classify_moneyness(100, 80) == "ITM"


def test_classify_moneyness_otm():
    assert classify_moneyness(100, 103.01) == "OTM"
    assert classify_moneyness(100, 120) == "OTM"


def test_label_regimes_adds_regime_column():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "vix_close": [12.0, 20.0, 30.0],
        "close": [100.0, 101.0, 102.0],
    })
    result = label_regimes(df)
    assert "regime" in result.columns
    assert list(result["regime"]) == ["low", "medium", "high"]


def test_label_regimes_does_not_modify_input():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "vix_close": [20.0],
        "close": [100.0],
    })
    original_cols = set(df.columns)
    label_regimes(df)
    assert set(df.columns) == original_cols

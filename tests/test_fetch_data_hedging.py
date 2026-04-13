import pandas as pd

import src.fetch_data_hedging_v2 as fetch_data_hedging_v2


def test_build_synthetic_market_dataset_accepts_rate_series(monkeypatch):
    stock_df = pd.DataFrame({
        "date": pd.to_datetime(["2025-03-31"]),
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.0],
        "volume": [1_000],
    })
    vix_df = pd.DataFrame({
        "date": pd.to_datetime(["2025-03-31"]),
        "vix_open": [20.0],
        "vix_high": [21.0],
        "vix_low": [19.0],
        "vix_close": [20.0],
        "vix_volume": [1_000],
    })
    rates_df = pd.DataFrame({
        "date": pd.to_datetime(["2025-03-31"]),
        "risk_free_rate_pct": [5.0],
    })
    merged_df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1_000],
            "vix_close": [20.0],
            "risk_free_rate_pct": [5.0],
            "risk_free_rate": [0.05],
            "dividend_yield": [0.01],
            "realized_vol_21d": [0.2],
            "base_iv": [0.2],
        },
        index=pd.to_datetime(["2025-03-31"]),
    )
    option_chain_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-03-31"]),
            "expiry": pd.to_datetime(["2025-04-30"]),
            "option_type": ["call"],
            "strike": [100.0],
        }
    )

    captured = {}

    def fake_fetch_market_inputs(ticker, months, end_date, rate_series, dividend_yield):
        captured["ticker"] = ticker
        captured["months"] = months
        captured["end_date"] = end_date
        captured["rate_series"] = rate_series
        captured["dividend_yield"] = dividend_yield
        return stock_df, vix_df, rates_df, merged_df

    def fake_simulate_daily_option_chain(ticker, merged_df, expiry_days):
        captured["expiry_days"] = expiry_days
        return option_chain_df

    monkeypatch.setattr(fetch_data_hedging_v2, "fetch_market_inputs", fake_fetch_market_inputs)
    monkeypatch.setattr(
        fetch_data_hedging_v2,
        "simulate_daily_option_chain",
        fake_simulate_daily_option_chain,
    )

    dataset = fetch_data_hedging_v2.build_synthetic_market_dataset(
        ticker="AAPL",
        months=6,
        end_date="2025-03-31",
        rate_series="DGS3MO",
        expiry_days=(30,),
        dividend_yield=0.01,
    )

    assert captured == {
        "ticker": "AAPL",
        "months": 6,
        "end_date": "2025-03-31",
        "rate_series": "DGS3MO",
        "expiry_days": (30,),
        "dividend_yield": 0.01,
    }
    assert dataset["stock_history"].equals(stock_df)
    assert dataset["vix_history"].equals(vix_df)
    assert dataset["risk_free_history"].equals(rates_df)
    assert dataset["synthetic_option_chain"].equals(option_chain_df)
    assert list(dataset["merged_daily_inputs"]["date"]) == [pd.Timestamp("2025-03-31")]
    assert list(dataset["merged_daily_inputs"]["dividend_yield"]) == [0.01]


def test_black_scholes_with_dividend_yield_reduces_call_delta_and_price():
    no_div_price = fetch_data_hedging_v2.black_scholes_price(
        100.0, 100.0, 30 / 365.0, 0.05, 0.20, "call", q=0.0
    )
    div_price = fetch_data_hedging_v2.black_scholes_price(
        100.0, 100.0, 30 / 365.0, 0.05, 0.20, "call", q=0.02
    )

    no_div_delta, *_ = fetch_data_hedging_v2.black_scholes_greeks(
        100.0, 100.0, 30 / 365.0, 0.05, 0.20, "call", q=0.0
    )
    div_delta, *_ = fetch_data_hedging_v2.black_scholes_greeks(
        100.0, 100.0, 30 / 365.0, 0.05, 0.20, "call", q=0.02
    )

    assert div_price < no_div_price
    assert div_delta < no_div_delta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime


# =========================================================
# Black-Scholes pricing
# =========================================================
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    S      : spot price
    K      : strike
    T      : time to expiry in years
    r      : annual risk-free rate in decimal form
    sigma  : annual volatility in decimal form
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Returns delta, gamma, vega, theta, rho.

    Notes:
    - vega is per 1.00 change in volatility
    - theta is annualized
    - rho is per 1.00 change in rates
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0

        return delta, 0.0, 0.0, 0.0, 0.0

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = norm.cdf(d1)
    nd1 = norm.pdf(d1)

    gamma = nd1 / (S * sigma * sqrtT)
    vega = S * nd1 * sqrtT

    if option_type == "call":
        delta = Nd1
        theta = -(S * nd1 * sigma) / (2.0 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = Nd1 - 1.0
        theta = -(S * nd1 * sigma) / (2.0 * sqrtT) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return delta, gamma, vega, theta, rho


# =========================================================
# Helpers
# =========================================================
def strike_step_from_spot(spot):
    """
    Simple synthetic strike grid step.
    """
    if spot < 25:
        return 1.0
    elif spot < 100:
        return 2.5
    elif spot < 250:
        return 5.0
    else:
        return 10.0


def build_strike_grid(spot, low_mult=0.80, high_mult=1.20):
    step = strike_step_from_spot(spot)
    low = np.floor((spot * low_mult) / step) * step
    high = np.ceil((spot * high_mult) / step) * step
    strikes = np.arange(low, high + step, step)
    return np.round(strikes, 2)


def normalize_ohlcv(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if isinstance(df.columns, pd.MultiIndex):
        expected = {"open", "high", "low", "close", "volume"}
        best_level = 0
        best_score = -1

        for level in range(df.columns.nlevels):
            level_values = {
                str(col).lower().replace(" ", "_")
                for col in df.columns.get_level_values(level)
            }
            score = len(expected & level_values)
            if score > best_score:
                best_score = score
                best_level = level

        df.columns = df.columns.get_level_values(best_level)

    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    expected = ["open", "high", "low", "close", "volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected OHLCV columns: {missing}")
    return df[expected]


def reset_index_with_date(df):
    out = df.reset_index().copy()
    first_col = out.columns[0]
    return out.rename(columns={first_col: "date"})


# =========================================================
# Data fetch
# =========================================================
def fetch_market_inputs(
    ticker: str,
    months: int = 3,
    end_date: str | None = None,
    # rate_series: str = "DGS3MO",
):
    """
    Fetch stock OHLCV, VIX, and FRED 3M Treasury rate for the last `months` months.
    """
    if end_date is None:
        end = pd.Timestamp.today().normalize()
    else:
        end = pd.Timestamp(end_date).normalize()

    start = end - pd.DateOffset(months=months)

    # Stock OHLCV
    stock = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if stock.empty:
        raise ValueError(f"No stock data returned for ticker: {ticker}")
    stock = normalize_ohlcv(stock)

    # VIX OHLCV
    vix = yf.download(
        "^VIX",
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if vix.empty:
        raise ValueError("No VIX data returned.")
    vix = normalize_ohlcv(vix)
    vix = vix.rename(columns={
        "open": "vix_open",
        "high": "vix_high",
        "low": "vix_low",
        "close": "vix_close",
        "volume": "vix_volume",
    })

    # FRED risk-free rate
    tbill = yf.download(
        "^IRX",
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )
    if tbill.empty:
        raise ValueError("No T-bill data returned from yfinance (^IRX).")
    tbill.index = pd.to_datetime(tbill.index).tz_localize(None)
    close_col = tbill["Close"]
    # yfinance >=0.2 returns MultiIndex columns; squeeze to Series if needed
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    rates = close_col.rename("risk_free_rate_pct").to_frame()

    # Align to stock trading days
    merged = stock.join(vix[["vix_close"]], how="left")
    merged = merged.join(rates, how="left")

    # Fill missing values to trading dates
    merged["vix_close"] = merged["vix_close"].ffill().bfill()
    merged["risk_free_rate_pct"] = merged["risk_free_rate_pct"].ffill().bfill()

    # Convert rate to decimal
    merged["risk_free_rate"] = merged["risk_free_rate_pct"] / 100.0

    # Realized vol fallback: 21-day rolling annualized vol
    log_ret = np.log(merged["close"] / merged["close"].shift(1))
    merged["realized_vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)

    # Base IV from VIX, fallback to realized vol if ever missing
    merged["base_iv"] = merged["vix_close"] / 100.0
    merged["base_iv"] = merged["base_iv"].fillna(merged["realized_vol_21d"])
    merged["base_iv"] = merged["base_iv"].clip(lower=0.05)

    stock_df = reset_index_with_date(stock)
    vix_df = reset_index_with_date(vix)
    rates_df = reset_index_with_date(rates)

    return stock_df, vix_df, rates_df, merged


# =========================================================
# Synthetic full daily option chain
# =========================================================
def simulate_daily_option_chain(
    ticker: str,
    merged_df: pd.DataFrame,
    expiry_days=(7, 14, 21, 30, 45, 60, 90),
    strike_low_mult=0.80,
    strike_high_mult=1.20,
    smile_strength=0.30,
    term_slope=0.08,
):
    """
    Create a synthetic option chain for each trading date.

    synthetic_iv formula:
        synthetic_iv = base_iv * (1 + smile_strength * abs(log(K/S))) * term_adjustment

    term_adjustment mildly changes IV by maturity relative to a 30-day anchor.
    """
    np.random.seed(42) 
    rows = []

    for trade_date, row in merged_df.iterrows():
        S = float(row["close"])
        r = float(row["risk_free_rate"])
        base_iv = float(row["base_iv"])

        if not np.isfinite(S) or S <= 0:
            continue

        strikes = build_strike_grid(S, low_mult=strike_low_mult, high_mult=strike_high_mult)

        for dte in expiry_days:
            expiry = trade_date + pd.Timedelta(days=int(dte))
            T = dte / 365.0

            # Mild term structure around 30-day anchor
            term_adjustment = 1.0 + term_slope * (np.sqrt(T) - np.sqrt(30 / 365.0))

            for K in strikes:
                log_moneyness = np.log(K / S)
                smile_adjustment = 1.0 + smile_strength * abs(log_moneyness)

                sigma = base_iv * smile_adjustment * term_adjustment
                noise = np.random.normal(0, 0.01)   # ±1% vol noise
                sigma = float(np.clip(sigma + noise, 0.05, 2.00))

                for option_type in ("call", "put"):
                    price = black_scholes_price(S, K, T, r, sigma, option_type)
                    delta, gamma, vega, theta, rho = black_scholes_greeks(
                        S, K, T, r, sigma, option_type
                    )

                    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
                    time_value = max(price - intrinsic, 0.0)

                    rows.append({
                        "trade_date": trade_date.date(),
                        "ticker": ticker,
                        "spot": round(S, 4),
                        "expiry": expiry.date(),
                        "dte": int(dte),
                        "strike": float(K),
                        "option_type": option_type,
                        "risk_free_rate": round(r, 8),
                        "vix_close": round(float(row["vix_close"]), 4),
                        "base_iv": round(base_iv, 6),
                        "synthetic_iv": round(sigma, 6),
                        "theoretical_price": round(price, 6),
                        "intrinsic_value": round(intrinsic, 6),
                        "time_value": round(time_value, 6),
                        "delta": round(delta, 6),
                        "gamma": round(gamma, 6),
                        "vega": round(vega, 6),
                        "theta": round(theta, 6),
                        "rho": round(rho, 6),
                    })

    option_chain_df = pd.DataFrame(rows)
    option_chain_df = option_chain_df.sort_values(
        ["trade_date", "expiry", "option_type", "strike"]
    ).reset_index(drop=True)

    return option_chain_df


# =========================================================
# Main wrapper
# =========================================================
def build_synthetic_market_dataset(
    ticker: str,
    months: int = 3,
    end_date: str | None = None,
    # rate_series: str = "DGS3MO",
    expiry_days=(7, 14, 21, 30, 45, 60, 90),
):
    """
    Returns:
        {
            'stock_history': ...,
            'vix_history': ...,
            'risk_free_history': ...,
            'merged_daily_inputs': ...,
            'synthetic_option_chain': ...
        }
    """
    stock_df, vix_df, rates_df, merged_df = fetch_market_inputs(
        ticker=ticker,
        months=months,
        end_date=end_date,
        # rate_series=rate_series,
    )

    option_chain_df = simulate_daily_option_chain(
        ticker=ticker,
        merged_df=merged_df,
        expiry_days=expiry_days,
    )

    merged_daily_inputs = reset_index_with_date(merged_df)

    return {
        "stock_history": stock_df,
        "vix_history": vix_df,
        "risk_free_history": rates_df,
        "merged_daily_inputs": merged_daily_inputs,
        "synthetic_option_chain": option_chain_df,
    }


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    dataset = build_synthetic_market_dataset(
        ticker="AAPL",      # change to SPY, MSFT, NVDA, etc.
        months=3,
        end_date=None,      # or "2026-03-31"
        # rate_series="DGS3MO",
        expiry_days=(7, 14, 21, 30, 45, 60, 90),
    )

    stock_df = dataset["stock_history"]
    vix_df = dataset["vix_history"]
    rates_df = dataset["risk_free_history"]
    merged_df = dataset["merged_daily_inputs"]
    option_chain_df = dataset["synthetic_option_chain"]

    print("\n=== STOCK HISTORY ===")
    print(stock_df.head())

    print("\n=== VIX HISTORY ===")
    print(vix_df[["date", "vix_close"]].head())

    print("\n=== RISK-FREE HISTORY ===")
    print(rates_df.head())

    print("\n=== MERGED DAILY INPUTS ===")
    print(merged_df[["date", "close", "vix_close", "risk_free_rate", "base_iv"]].head())

    print("\n=== SYNTHETIC OPTION CHAIN ===")
    print(option_chain_df.head(20))

    # Optional saves
    stock_df.to_csv("stock_history.csv", index=False)
    vix_df.to_csv("vix_history.csv", index=False)
    rates_df.to_csv("risk_free_history.csv", index=False)
    merged_df.to_csv("merged_daily_inputs.csv", index=False)
    option_chain_df.to_csv("synthetic_option_chain.csv", index=False)

    print("\nSaved CSV files to current directory.")
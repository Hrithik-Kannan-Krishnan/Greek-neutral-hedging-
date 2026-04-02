from __future__ import annotations

import math
from typing import Tuple


def bs_call_price_greeks(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[float, float, float, float, float]:
    """Compute Black-Scholes price and Greeks for a European call.

    Returns (price, delta, gamma, vega, theta) as a 5-float tuple.
    Handles degenerate cases (T<=0, sigma<=0, S<=0, K<=0) via intrinsic value.
    """
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)

    # Degenerate / edge case
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        price = float(max(S - K, 0.0))
        delta = 1.0 if S > K else 0.0
        return (price, delta, 0.0, 0.0, 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Standard normal CDF and PDF
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    n = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    price = S * N(d1) - K * math.exp(-r * T) * N(d2)
    delta = N(d1)
    gamma = n(d1) / (S * sigma * sqrt_T)
    vega = S * n(d1) * sqrt_T
    theta = -(S * n(d1) * sigma) / (2.0 * sqrt_T) - r * K * math.exp(-r * T) * N(d2)

    return (price, delta, gamma, vega, theta)


def bs_call_price_delta(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[float, float]:
    """Return only (price, delta) for a European call via Black-Scholes.

    Delegates to bs_call_price_greeks and returns the first two elements.
    """
    result = bs_call_price_greeks(S, K, T, r, sigma)
    return (result[0], result[1])


def heston_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    v0: float = 0.04,
) -> Tuple[float, float, float, float, float]:
    """Compute Heston model price and Greeks for a European call via QuantLib.

    Returns (price, delta, gamma, vega, theta_greek). Falls back to finite
    differences if analytical Greeks raise an exception.
    """
    try:
        import QuantLib as ql
    except ImportError:
        raise NotImplementedError(
            "QuantLib is required for Heston pricing. Install with: pip install QuantLib"
        )

    S, K, T, r = float(S), float(K), float(T), float(r)
    kappa, theta, sigma, rho, v0 = (
        float(kappa), float(theta), float(sigma), float(rho), float(v0)
    )

    # Edge case
    if T <= 0.0 or S <= 0.0 or K <= 0.0:
        price = float(max(S - K, 0.0))
        delta = 1.0 if S > K else 0.0
        return (price, delta, 0.0, 0.0, 0.0)

    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))

    process = ql.HestonProcess(rate_ts, div_ts, spot_handle, v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    maturity_date = today + ql.Period(max(1, int(round(T * 365))), ql.Days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)

    price = float(option.NPV())

    # Try analytical Greeks, fall back to finite differences
    try:
        delta_greek = float(option.delta())
        gamma_greek = float(option.gamma())
        vega_greek = float(option.vega())
        theta_greek = float(option.thetaPerDay()) * 365.0
    except Exception:
        # Finite-difference fallback
        h = 0.01
        spot_up = ql.SimpleQuote(S + h)
        spot_dn = ql.SimpleQuote(S - h)

        def _price_with_spot(s_val: float) -> float:
            sq = ql.SimpleQuote(s_val)
            sh = ql.QuoteHandle(sq)
            proc = ql.HestonProcess(rate_ts, div_ts, sh, v0, kappa, theta, sigma, rho)
            mdl = ql.HestonModel(proc)
            eng = ql.AnalyticHestonEngine(mdl)
            opt = ql.VanillaOption(payoff, exercise)
            opt.setPricingEngine(eng)
            return float(opt.NPV())

        p_up = _price_with_spot(S + h)
        p_dn = _price_with_spot(S - h)
        delta_greek = (p_up - p_dn) / (2.0 * h)
        gamma_greek = (p_up - 2.0 * price + p_dn) / (h * h)

        # Vega: bump v0
        dv = 0.001

        def _price_with_v0(v0_val: float) -> float:
            proc = ql.HestonProcess(rate_ts, div_ts, spot_handle, v0_val, kappa, theta, sigma, rho)
            mdl = ql.HestonModel(proc)
            eng = ql.AnalyticHestonEngine(mdl)
            opt = ql.VanillaOption(payoff, exercise)
            opt.setPricingEngine(eng)
            return float(opt.NPV())

        vega_greek = (_price_with_v0(v0 + dv) - _price_with_v0(v0 - dv)) / (2.0 * dv)

        # Theta fallback via BS
        _, _, _, _, theta_greek = bs_call_price_greeks(S, K, T, r, math.sqrt(v0))

    return (price, delta_greek, gamma_greek, vega_greek, theta_greek)

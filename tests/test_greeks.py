# tests/test_greeks.py
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.greeks import bs_call_price_greeks, bs_call_price_delta


def test_bs_call_price_greeks_atm_known_values():
    # ATM call: S=100, K=100, T=1, r=0.05, sigma=0.20
    # Known BS price ≈ 10.45, delta ≈ 0.637
    price, delta, gamma, vega, theta = bs_call_price_greeks(100, 100, 1.0, 0.05, 0.20)
    assert abs(price - 10.4506) < 0.01
    assert abs(delta - 0.6368) < 0.001
    assert gamma > 0
    assert vega > 0
    assert theta < 0  # theta is negative for long options


def test_bs_call_price_greeks_returns_five_floats():
    result = bs_call_price_greeks(150, 155, 0.25, 0.04, 0.25)
    assert len(result) == 5
    assert all(isinstance(v, float) for v in result)


def test_bs_call_price_greeks_expiry_zero_itm():
    price, delta, gamma, vega, theta = bs_call_price_greeks(105, 100, 0.0, 0.05, 0.20)
    assert abs(price - 5.0) < 1e-9
    assert delta == 1.0


def test_bs_call_price_greeks_expiry_zero_otm():
    price, delta, gamma, vega, theta = bs_call_price_greeks(95, 100, 0.0, 0.05, 0.20)
    assert price == 0.0
    assert delta == 0.0


def test_bs_call_price_delta_matches_greeks():
    S, K, T, r, sigma = 120, 115, 0.5, 0.03, 0.30
    price_d, delta_d = bs_call_price_delta(S, K, T, r, sigma)
    price_g, delta_g, *_ = bs_call_price_greeks(S, K, T, r, sigma)
    assert abs(price_d - price_g) < 1e-10
    assert abs(delta_d - delta_g) < 1e-10


def test_bs_call_price_greeks_expiry_zero_atm():
    # ATM at expiry: S == K, intrinsic = 0, delta convention = 0.5
    price, delta, gamma, vega, theta = bs_call_price_greeks(100, 100, 0.0, 0.05, 0.20)
    assert price == 0.0
    assert delta == 0.5


def test_bs_call_price_greeks_gamma_and_vega_numerical():
    # ATM call: S=100, K=100, T=1, r=0.05, sigma=0.20
    # Exact: gamma=0.018762, vega=37.524035
    price, delta, gamma, vega, theta = bs_call_price_greeks(100, 100, 1.0, 0.05, 0.20)
    assert abs(gamma - 0.018762) < 0.0005  # tight tolerance on exact value
    assert abs(vega - 37.524) < 0.5        # per unit sigma; tight on exact value


def test_heston_greeks_raises_if_quantlib_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "QuantLib":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", mock_import)
    from src import greeks as g
    import importlib
    importlib.reload(g)
    with pytest.raises(NotImplementedError, match="QuantLib"):
        g.heston_greeks(100, 100, 1.0, 0.05)

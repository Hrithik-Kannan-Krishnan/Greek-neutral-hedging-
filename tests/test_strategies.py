# tests/test_strategies.py
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategies import (
    delta_neutral,
    delta_gamma_neutral,
    delta_vega_neutral,
    delta_theta_neutral,
)

# Convenience: (price, delta, gamma, vega, theta)
PRIMARY = (5.0, 0.50, 0.02, 10.0, -0.05)
HEDGE   = (3.0, 0.30, 0.015, 8.0, -0.04)


def test_delta_neutral_returns_zero_hedge():
    stock, hedge = delta_neutral(PRIMARY, n_primary=10, contract_multiplier=100)
    assert hedge == 0
    assert abs(stock - (-10 * 100 * 0.50)) < 1e-6


def test_delta_neutral_rounds_stock_to_whole_number():
    stock, hedge = delta_neutral((5.0, 0.333, 0.01, 5.0, -0.02), n_primary=3, contract_multiplier=100)
    assert stock == int(stock)


def test_delta_gamma_neutral_gamma_balanced():
    stock, hedge = delta_gamma_neutral(PRIMARY, HEDGE, n_primary=10, contract_multiplier=100)
    expected_hc = round(-10 * 0.02 / 0.015)
    assert hedge == expected_hc


def test_delta_gamma_neutral_delta_balanced():
    stock, hedge = delta_gamma_neutral(PRIMARY, HEDGE, n_primary=10, contract_multiplier=100)
    hc = round(-10 * 0.02 / 0.015)
    expected_stock = round(-(10 * 100 * 0.50 + hc * 100 * 0.30))
    assert stock == expected_stock


def test_delta_vega_neutral():
    stock, hedge = delta_vega_neutral(PRIMARY, HEDGE, n_primary=10, contract_multiplier=100)
    expected_hc = round(-10 * 10.0 / 8.0)
    assert hedge == expected_hc


def test_delta_theta_neutral():
    stock, hedge = delta_theta_neutral(PRIMARY, HEDGE, n_primary=10, contract_multiplier=100)
    expected_hc = round(-10 * (-0.05) / (-0.04))
    assert hedge == expected_hc


def test_near_zero_denominator_returns_zero_hedge():
    hedge_zero_gamma = (3.0, 0.30, 0.0, 8.0, -0.04)
    stock, hedge = delta_gamma_neutral(PRIMARY, hedge_zero_gamma, n_primary=10, contract_multiplier=100)
    assert hedge == 0


def test_all_strategies_return_int_positions():
    for fn in [delta_gamma_neutral, delta_vega_neutral, delta_theta_neutral]:
        stock, hedge = fn(PRIMARY, HEDGE, n_primary=10, contract_multiplier=100)
        assert stock == int(stock)
        assert hedge == int(hedge)

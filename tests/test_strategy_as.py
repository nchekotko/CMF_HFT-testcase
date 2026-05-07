from __future__ import annotations

import math

from cmf_mm.lob.book import OrderBookState
from cmf_mm.strategy.as_microprice import ASMicropriceStrategy
from cmf_mm.strategy.avellaneda_stoikov import (
    ASParams,
    AvellanedaStoikovStrategy,
    optimal_spread,
    reservation_price,
)
from cmf_mm.strategy.base import StrategyState
from cmf_mm.types import LOBEvent


def test_reservation_price_zero_inventory_equals_mid():
    s, q, gamma, sigma, dt = 100.0, 0.0, 1.0, 0.5, 1.0
    assert reservation_price(s, q, gamma, sigma, dt) == s


def test_reservation_price_long_inventory_pulls_below_mid():
    s, q, gamma, sigma, dt = 100.0, 5.0, 1.0, 0.5, 1.0
    r = reservation_price(s, q, gamma, sigma, dt)
    assert r < s
    assert math.isclose(r, s - q * gamma * sigma * sigma * dt)


def test_optimal_spread_positive():
    """Optimal spread is bounded below by 1/k for small γ and grows linearly
    in γ once the inventory-penalty term γσ²dt dominates."""
    assert optimal_spread(0.5, 0.5, 1.0, 1.0) > 0
    assert optimal_spread(2.0, 0.5, 1.0, 1.0) > 0
    # For very large γ, the inventory penalty dominates; the spread grows
    # without bound. Assert the asymptotic regime.
    sp_small = optimal_spread(0.1, 0.5, 1.0, 1.0)
    sp_large = optimal_spread(100.0, 0.5, 1.0, 1.0)
    assert sp_large > sp_small


def test_strategy_emits_two_quotes_when_book_initialised():
    book = OrderBookState()
    book.update(LOBEvent(ts=0, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0))
    p = ASParams(gamma=0.5, sigma=0.1, k=1.0, T_horizon_us=0,
                 order_size=1.0, max_inventory=10.0,
                 quote_refresh_min_interval_us=0)
    strat = AvellanedaStoikovStrategy(p)
    out = strat.on_event(book, LOBEvent(ts=1, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0),
                        StrategyState())
    assert out is not None
    sides = {a.side for a in out}
    assert sides == {"buy", "sell"}


def test_microprice_variant_uses_microprice_as_reference():
    book = OrderBookState()
    # Imbalanced book: bid heavy ⇒ microprice > mid
    book.update(LOBEvent(ts=0, bid_px=99.0, bid_sz=20.0, ask_px=101.0, ask_sz=2.0))
    p = ASParams(gamma=0.5, sigma=0.0, k=1.0, T_horizon_us=0,
                 order_size=1.0, max_inventory=10.0,
                 quote_refresh_min_interval_us=0)
    v1 = AvellanedaStoikovStrategy(p)
    v2 = ASMicropriceStrategy(p)
    out_v1 = v1.on_event(book, LOBEvent(ts=1, bid_px=99.0, bid_sz=20.0, ask_px=101.0, ask_sz=2.0),
                         StrategyState())
    out_v2 = v2.on_event(book, LOBEvent(ts=1, bid_px=99.0, bid_sz=20.0, ask_px=101.0, ask_sz=2.0),
                         StrategyState())
    assert out_v1 is not None and out_v2 is not None
    # σ=0 ⇒ inventory penalty vanishes and quotes are symmetric around the
    # reference. Reference for V2 is microprice > mid, so V2's bid sits above V1's.
    bid_v1 = next(a.price for a in out_v1 if a.side == "buy")
    bid_v2 = next(a.price for a in out_v2 if a.side == "buy")
    assert bid_v2 > bid_v1


def test_throttle_returns_none_within_min_interval():
    book = OrderBookState()
    book.update(LOBEvent(ts=0, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0))
    p = ASParams(gamma=0.5, sigma=0.1, k=1.0, T_horizon_us=0,
                 order_size=1.0, max_inventory=10.0,
                 quote_refresh_min_interval_us=1_000_000)
    strat = AvellanedaStoikovStrategy(p)
    state = StrategyState()
    e1 = LOBEvent(ts=10_000, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0)
    e2 = LOBEvent(ts=20_000, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0)
    out1 = strat.on_event(book, e1, state)
    out2 = strat.on_event(book, e2, state)
    assert out1 is not None
    assert out2 is None

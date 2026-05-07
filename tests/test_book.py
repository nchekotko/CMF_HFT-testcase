from __future__ import annotations

import math

from cmf_mm.lob.book import OrderBookState
from cmf_mm.lob.microprice import weighted_mid
from cmf_mm.types import LOBEvent


def test_book_initial_state_uninitialised():
    b = OrderBookState()
    assert not b.is_initialised()


def test_book_update_sets_top_of_book(tiny_book_event):
    b = OrderBookState()
    b.update(tiny_book_event)
    assert b.is_initialised()
    assert b.bid_px == 99.5
    assert b.ask_px == 100.5
    assert math.isclose(b.mid(), 100.0)
    assert math.isclose(b.spread(), 1.0)


def test_book_imbalance_balanced():
    b = OrderBookState()
    b.update(LOBEvent(ts=0, bid_px=1.0, bid_sz=5.0, ask_px=2.0, ask_sz=5.0))
    assert b.imbalance() == 0.0


def test_book_imbalance_buy_heavy():
    b = OrderBookState()
    b.update(LOBEvent(ts=0, bid_px=1.0, bid_sz=10.0, ask_px=2.0, ask_sz=2.0))
    # I = (10 - 2) / (10 + 2) ≈ 0.6667
    assert math.isclose(b.imbalance(), 8.0 / 12.0)


def test_book_imbalance_zero_depth_returns_zero():
    b = OrderBookState()
    b.update(LOBEvent(ts=0, bid_px=1.0, bid_sz=0.0, ask_px=2.0, ask_sz=0.0))
    assert b.imbalance() == 0.0


def test_microprice_buy_heavy_pushes_up():
    """When bids are deeper, micro-price should sit closer to ask."""
    b = OrderBookState()
    b.update(LOBEvent(ts=0, bid_px=99.0, bid_sz=10.0, ask_px=101.0, ask_sz=2.0))
    micro = weighted_mid(b)
    # p_micro = (10*101 + 2*99)/(2+10) = (1010 + 198)/12 = 100.666...
    assert math.isclose(micro, (10 * 101 + 2 * 99) / 12.0)
    assert micro > b.mid()


def test_microprice_falls_back_to_mid_when_zero_depth():
    b = OrderBookState()
    b.update(LOBEvent(ts=0, bid_px=99.0, bid_sz=0.0, ask_px=101.0, ask_sz=0.0))
    assert weighted_mid(b) == b.mid()

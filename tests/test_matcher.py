"""Execution rule tests — these are the most important invariants in the engine."""

from __future__ import annotations

from cmf_mm.engine.matcher import check_fills
from cmf_mm.types import Order, OrderId, TradeEvent


def _o(side, price, size, oid=1, submit_ts=0):
    return Order(order_id=OrderId(oid), side=side, price=price, size=size, submit_ts=submit_ts)


def test_buy_limit_filled_when_trade_at_or_below():
    order = _o("buy", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.0)
    assert len(fills) == 1
    assert fills[0].price == 100.0
    assert fills[0].side == "buy"


def test_buy_limit_not_filled_when_trade_above():
    order = _o("buy", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=100.5, size=1.0, aggressor_side="buy")
    assert check_fills(trade, [order], mid_at_fill=100.0) == []


def test_sell_limit_filled_when_trade_at_or_above():
    order = _o("sell", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=100.5, size=1.0, aggressor_side="buy")
    fills = check_fills(trade, [order], mid_at_fill=100.0)
    assert len(fills) == 1
    assert fills[0].price == 100.0


def test_sell_limit_not_filled_when_trade_below():
    order = _o("sell", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="sell")
    assert check_fills(trade, [order], mid_at_fill=100.0) == []


def test_aggressor_must_match_side():
    """A buy-aggressor trade through our resting buy means a different
    counter-party — should not fill our buy."""
    order = _o("buy", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="buy")
    assert check_fills(trade, [order], mid_at_fill=100.0) == []


def test_fifo_order_for_same_side_at_same_price():
    o1 = _o("buy", 100.0, 1.0, oid=1, submit_ts=10)
    o2 = _o("buy", 100.0, 1.0, oid=2, submit_ts=20)
    trade = TradeEvent(ts=30, price=99.5, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [o2, o1], mid_at_fill=100.0)
    # Only one trade size unit available, FIFO should give it to o1
    assert [f.order_id for f in fills] == [OrderId(1)]


def test_no_partial_fills_by_default():
    order = _o("buy", 100.0, 5.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.0, partial_fills=False)
    assert fills == []


def test_partial_fills_when_enabled():
    order = _o("buy", 100.0, 5.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.0, partial_fills=True)
    assert len(fills) == 1
    assert fills[0].size == 1.0

"""Execution rule: a resting limit fills when a market trade crosses it.

Conventions:
  - A resting BUY @ p fills when an incoming sell-aggressor trade prints at
    a price ≤ p.
  - A resting SELL @ p fills when an incoming buy-aggressor trade prints at
    a price ≥ p.
  - Fills happen at the resting limit's price, not at the trade's print
    price (the maker captures the spread).
  - Multiple resting orders crossed by the same trade fill in FIFO order
    (earliest submit_ts first), up to the trade's printed size.
  - Default behaviour is no partial fills: an order either fills its full
    size or not at all. This is conservative for an MM backtest because it
    avoids ambiguity around queue position, which we do not model.

This module is intentionally pure — it consumes a trade and a list of
active orders, returns the fills it produced and the order ids it consumed.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..types import Fill, Order, TradeEvent


def _crossed(order: Order, trade: TradeEvent) -> bool:
    if order.side == "buy":
        # filled by sell-aggressor printing at or below our bid
        return trade.aggressor_side == "sell" and trade.price <= order.price
    else:
        return trade.aggressor_side == "buy" and trade.price >= order.price


def check_fills(
    trade: TradeEvent,
    active_orders: Iterable[Order],
    mid_at_fill: float,
    *,
    partial_fills: bool = False,
) -> list[Fill]:
    """Return fills produced against ``active_orders`` by ``trade``.

    ``mid_at_fill`` is the engine-tracked mid immediately before the trade,
    used for spread-capture decomposition downstream.
    """
    candidates = sorted([o for o in active_orders if _crossed(o, trade)], key=lambda o: o.submit_ts)
    if not candidates:
        return []

    fills: list[Fill] = []
    remaining = trade.size
    for o in candidates:
        if remaining <= 0:
            break
        if partial_fills:
            qty = min(o.size, remaining)
        else:
            if o.size > remaining:
                # not enough trade size to fill this order fully — skip
                continue
            qty = o.size
        fills.append(
            Fill(
                order_id=o.order_id,
                ts=trade.ts,
                side=o.side,
                price=o.price,
                size=qty,
                mid_at_fill=mid_at_fill,
            )
        )
        remaining -= qty
    return fills

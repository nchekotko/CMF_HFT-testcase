"""Active-order tracking and quote diffing.

Strategies emit a *desired* quote state (a list of ``QuoteAction``); the
order manager diffs that against currently-active orders and issues the
necessary cancel/place pairs. This keeps strategy code declarative.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast, get_args

from ..types import Order, OrderId, QuoteAction, Side


@dataclass(slots=True)
class OrderManager:
    _next_id: int = 1
    _orders: dict[OrderId, Order] = field(default_factory=dict)

    @property
    def active_orders(self) -> dict[OrderId, Order]:
        return self._orders

    def place(self, side: Side, price: float, size: float, ts: int) -> OrderId:
        oid = OrderId(self._next_id)
        self._next_id += 1
        self._orders[oid] = Order(order_id=oid, side=side, price=price, size=size, submit_ts=ts)
        return oid

    def cancel(self, order_id: OrderId) -> None:
        self._orders.pop(order_id, None)

    def cancel_side(self, side: Side) -> None:
        for oid in [oid for oid, o in self._orders.items() if o.side == side]:
            del self._orders[oid]

    def cancel_all(self) -> None:
        self._orders.clear()

    def remove(self, order_id: OrderId) -> None:
        """Drop an order without an explicit cancel (used after a fill)."""
        self._orders.pop(order_id, None)

    def reconcile(self, actions: Iterable[QuoteAction], ts: int) -> None:
        """Bring active orders in line with the desired quote state.

        Algorithm: for each side, if the desired (price, size) matches the
        current order (after a cancel of stale ones), keep it; otherwise
        cancel and replace. We require at most one active order per side,
        which is the case for AS-style market making.
        """
        desired: dict[Side, QuoteAction] = {}
        for a in actions:
            desired[a.side] = a

        # iterate the literal sides explicitly to keep mypy happy
        sides: tuple[Side, ...] = cast("tuple[Side, ...]", get_args(Side))
        for side in sides:
            cur = next((o for o in self._orders.values() if o.side == side), None)
            want = desired.get(side)
            if want is None:
                if cur is not None:
                    self.cancel(cur.order_id)
                continue
            if cur is None:
                self.place(side, want.price, want.size, ts)
            elif cur.price != want.price or cur.size != want.size:
                self.cancel(cur.order_id)
                self.place(side, want.price, want.size, ts)

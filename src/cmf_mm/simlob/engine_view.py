"""EngineView — one engine's private *diff* over the shared HistoricalLOB.

Each trading engine deviates from the real market in only two small ways, so we
store only those deltas (never a full copy of the book):

1. **Own resting orders** — synthetic orders this engine injected that are still
   live. Tracked per order so we can model queue position and cancel them.
2. **Consumed historical liquidity** — volume this engine's own aggressive
   trading already ate out of the real book. From this engine's perspective that
   liquidity is gone (it "doesn't see some orders if it already consumed them"),
   even though it is still present for every other engine and in HistoricalLOB.

An EngineView is owned by exactly one engine thread and mutated only by it, so it
needs no synchronisation. It is read alongside an immutable HistoricalLOB
snapshot by :class:`~cmf_mm.simlob.simulated.SimulatedLOB`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import OrderId, Side


@dataclass(slots=True)
class OwnOrder:
    """One synthetic order this engine has resting in its private view."""

    order_id: OrderId
    side: Side
    price: float
    size: float
    submit_ts: int
    # Historical volume resting at this price when we joined the queue: under a
    # FIFO model this much real liquidity must trade before our passive order
    # fills. Lets us go beyond fill-at-touch later without schema changes.
    volume_ahead: float = 0.0


@dataclass(slots=True)
class EngineView:
    engine_id: int
    _own: dict[OrderId, OwnOrder] = field(default_factory=dict)
    # (side, price) -> historical volume this engine has consumed at that level.
    _consumed: dict[tuple[Side, float], float] = field(default_factory=dict)

    # ---- own orders ------------------------------------------------------

    def add_own(self, order: OwnOrder) -> None:
        self._own[order.order_id] = order

    def cancel_own(self, order_id: OrderId) -> OwnOrder | None:
        return self._own.pop(order_id, None)

    def reduce_own(self, order_id: OrderId, qty: float) -> None:
        """Reduce a resting own order after a (partial) fill; drop if emptied."""
        order = self._own.get(order_id)
        if order is None:
            return
        order.size -= qty
        if order.size <= 1e-12:
            self._own.pop(order_id, None)

    def own_orders(self) -> list[OwnOrder]:
        return list(self._own.values())

    def own_size_at(self, side: Side, price: float) -> float:
        return sum(
            o.size for o in self._own.values() if o.side == side and o.price == price
        )

    # ---- consumed historical liquidity ----------------------------------

    def consume_historical(self, side: Side, price: float, qty: float) -> None:
        key = (side, price)
        self._consumed[key] = self._consumed.get(key, 0.0) + qty

    def consumed_at(self, side: Side, price: float) -> float:
        return self._consumed.get((side, price), 0.0)

    def consumed_levels(self) -> dict[tuple[Side, float], float]:
        """All ``(side, price) -> consumed volume`` entries (read-only copy)."""
        return dict(self._consumed)

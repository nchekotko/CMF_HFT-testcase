"""Canonical L3 (order-by-order) event model.

The production data source (Databento MBO / "market-by-order") delivers a stream
of order-level events keyed by ``order_id``. Everything in :mod:`simlob` is built
on top of this event type so that the core never depends on a particular vendor
schema: source adapters translate their native format into ``L3Event`` and the
rest of the system is identical regardless of where the data came from.

Timestamps are int microseconds since epoch (μs), matching the rest of the
backtester (see :mod:`cmf_mm.types`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..types import OrderId, Side


class L3Action(Enum):
    """Order-book mutation kinds in an MBO stream.

    ADD     a new resting order enters the book at (side, price, size)
    MODIFY  an existing order changes size (price change == CANCEL + ADD upstream)
    CANCEL  an existing order leaves the book before execution
    TRADE   an aggressor executes against a resting order (removes liquidity)
    CLEAR   the whole book is reset (session start / gap recovery)
    """

    ADD = "add"
    MODIFY = "modify"
    CANCEL = "cancel"
    TRADE = "trade"
    CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class L3Event:
    """One order-level event from the *historical* (real-market) stream.

    For ``TRADE`` events ``order_id`` identifies the *resting* order that was
    hit; ``size`` is the executed quantity. For ``ADD`` it is the new order.
    ``price``/``side`` may be unset (NaN / None) for ``MODIFY``/``CANCEL``/
    ``TRADE`` where the engine looks the order up by id, but adapters are
    encouraged to fill them in when known — it makes replay self-checking.
    """

    ts: int
    action: L3Action
    order_id: OrderId
    side: Side | None = None
    price: float = float("nan")
    size: float = 0.0

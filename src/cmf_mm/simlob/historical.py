"""HistoricalLOB — the shared "basement" book reconstructed from L3 replay.

This is the single source of truth that *all* engines read from. It is mutated
by exactly one writer (the backtest/replay thread) and never by an engine. To
let N engine threads read it concurrently without locks, the writer never hands
out its mutable internals: after applying a batch of events it *publishes* an
immutable :class:`BookSnapshot`. Engines read the latest snapshot reference,
which in CPython is an atomic pointer load (the GIL makes a single attribute
read/assign indivisible). The previous snapshot stays valid for any reader still
holding it — classic copy-on-write publication, no torn reads.

Memory model rationale (the assignment's key question): we keep **one** book and
publish small immutable snapshots, rather than N mutable copies. Each snapshot
shares nothing mutable with the live book, so a reader's view is stable even as
the writer races ahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..types import OrderId, Side
from .events import L3Action, L3Event


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Immutable aggregated view of the historical book at one sequence point.

    Levels are aggregated price -> total resting size (queue identity within a
    level is summarised to volume; that is all the fill model needs for
    fill-at-touch and for the FIFO "volume ahead" estimate). Sorted tuples make
    best-price access and level scans cheap and allocation-free for readers.
    """

    seq: int
    ts: int
    # bids: descending by price; asks: ascending by price.
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]

    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else math.nan

    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else math.nan

    def best_bid_size(self) -> float:
        return self.bids[0][1] if self.bids else 0.0

    def best_ask_size(self) -> float:
        return self.asks[0][1] if self.asks else 0.0

    def mid(self) -> float:
        return 0.5 * (self.best_bid() + self.best_ask())

    def volume_at(self, side: Side, price: float) -> float:
        """Total historical resting volume at an exact price level (0 if none)."""
        levels = self.bids if side == "buy" else self.asks
        for px, sz in levels:
            if px == price:
                return sz
        return 0.0


@dataclass(slots=True)
class _RestingOrder:
    side: Side
    price: float
    size: float


@dataclass(slots=True)
class HistoricalLOB:
    """Mutable, single-writer reconstruction of the real market book.

    Drive it with :meth:`apply` (one L3 event) then call :meth:`publish` to make
    a fresh immutable snapshot visible to readers via :attr:`latest`.
    """

    _bids: dict[float, float] = field(default_factory=dict)  # price -> volume
    _asks: dict[float, float] = field(default_factory=dict)
    _orders: dict[OrderId, _RestingOrder] = field(default_factory=dict)
    _seq: int = 0
    _ts: int = 0
    # Published immutable snapshot; readers load this reference atomically.
    latest: BookSnapshot = field(
        default_factory=lambda: BookSnapshot(0, 0, (), ())
    )

    # ---- writer side -----------------------------------------------------

    def apply(self, event: L3Event) -> None:
        """Fold one L3 event into the book. Does not publish (batch then publish)."""
        self._ts = event.ts
        action = event.action
        if action is L3Action.ADD:
            self._add(event)
        elif action is L3Action.CANCEL:
            self._remove(event.order_id, event.size, full=True)
        elif action is L3Action.MODIFY:
            self._modify(event)
        elif action is L3Action.TRADE:
            self._remove(event.order_id, event.size, full=False)
        elif action is L3Action.CLEAR:
            self._bids.clear()
            self._asks.clear()
            self._orders.clear()

    def _book(self, side: Side) -> dict[float, float]:
        return self._bids if side == "buy" else self._asks

    def _add(self, event: L3Event) -> None:
        if event.side is None or math.isnan(event.price):
            raise ValueError(f"ADD requires side and price: {event!r}")
        self._orders[event.order_id] = _RestingOrder(
            event.side, event.price, event.size
        )
        book = self._book(event.side)
        book[event.price] = book.get(event.price, 0.0) + event.size

    def _remove(self, order_id: OrderId, size: float, *, full: bool) -> None:
        order = self._orders.get(order_id)
        if order is None:
            return  # unknown id (e.g. order added before replay window) — ignore
        qty = order.size if full else min(size, order.size)
        order.size -= qty
        book = self._book(order.side)
        remaining_level = book.get(order.price, 0.0) - qty
        if remaining_level <= 1e-12:
            book.pop(order.price, None)
        else:
            book[order.price] = remaining_level
        if order.size <= 1e-12:
            self._orders.pop(order_id, None)

    def _modify(self, event: L3Event) -> None:
        order = self._orders.get(event.order_id)
        if order is None:
            return
        delta = event.size - order.size
        order.size = event.size
        book = self._book(order.side)
        new_level = book.get(order.price, 0.0) + delta
        if new_level <= 1e-12:
            book.pop(order.price, None)
        else:
            book[order.price] = new_level

    def publish(self) -> BookSnapshot:
        """Freeze current state into a new immutable snapshot and expose it.

        The reference assignment to :attr:`latest` is the publication point;
        readers that loaded the old reference keep a consistent view.
        """
        self._seq += 1
        bids = tuple(sorted(self._bids.items(), key=lambda kv: -kv[0]))
        asks = tuple(sorted(self._asks.items(), key=lambda kv: kv[0]))
        snap = BookSnapshot(self._seq, self._ts, bids, asks)
        self.latest = snap  # atomic pointer publish (CPython GIL)
        return snap

    def apply_and_publish(self, event: L3Event) -> BookSnapshot:
        self.apply(event)
        return self.publish()

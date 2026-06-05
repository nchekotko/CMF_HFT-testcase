"""SimulatedLOB — the book an engine *sees* = HistoricalLOB ⊕ EngineView.

This is a thin merge view, computed lazily on read. It never materialises a
private copy of the book: it reads an immutable :class:`BookSnapshot` plus the
engine's small overlay and combines them on demand. Effective volume at a level::

    effective(side, px) = max(0, historical(side, px) - consumed(side, px))
                          + own_size(side, px)

i.e. start from the real market, subtract what this engine already ate, add this
engine's own resting orders.

:class:`FillSimulator` drives synthetic execution against this view using the
simplest model the assignment asks for — **fill at touch**: a limit order that
is at or through the best historical price executes immediately at the touch
price. Resting orders are re-checked on every book advance so a passive quote
fills once the market trades to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import Fill, OrderId, Side
from .engine_view import EngineView, OwnOrder
from .historical import BookSnapshot, HistoricalLOB


@dataclass(slots=True)
class SimulatedLOB:
    """Read-only merged view for one engine. Cheap to construct per query."""

    snapshot: BookSnapshot
    view: EngineView

    def _effective_levels(self, side: Side) -> list[tuple[float, float]]:
        hist = self.snapshot.bids if side == "buy" else self.snapshot.asks
        prices: dict[float, float] = {}
        for px, sz in hist:
            eff = sz - self.view.consumed_at(side, px)
            if eff > 1e-12:
                prices[px] = eff
        for order in self.view.own_orders():
            if order.side == side:
                prices[order.price] = prices.get(order.price, 0.0) + order.size
        items = sorted(
            prices.items(), key=lambda kv: -kv[0] if side == "buy" else kv[0]
        )
        return items

    def bids(self) -> list[tuple[float, float]]:
        return self._effective_levels("buy")

    def asks(self) -> list[tuple[float, float]]:
        return self._effective_levels("sell")

    def best_bid(self) -> float:
        levels = self._effective_levels("buy")
        return levels[0][0] if levels else math.nan

    def best_ask(self) -> float:
        levels = self._effective_levels("sell")
        return levels[0][0] if levels else math.nan

    def mid(self) -> float:
        return 0.5 * (self.best_bid() + self.best_ask())


class FillSimulator:
    """Fill-at-touch execution against the historical book for one engine.

    Holds the shared (read-only) :class:`HistoricalLOB` and the engine's private
    :class:`EngineView`. Submitting an order may fill immediately (marketable);
    otherwise it rests and is filled later by :meth:`check_resting_fills`.
    """

    def __init__(self, historical: HistoricalLOB, view: EngineView) -> None:
        self._hist = historical
        self._view = view

    def view_now(self) -> SimulatedLOB:
        """The book this engine currently sees (latest published snapshot)."""
        return SimulatedLOB(self._hist.latest, self._view)

    def cancel(self, order_id: OrderId) -> bool:
        """Cancel a resting own order; True if it was live. Keeps the diff small
        for engines that continuously re-quote (cancel/replace)."""
        return self._view.cancel_own(order_id) is not None

    def submit_limit(
        self, order_id: OrderId, side: Side, price: float, size: float, ts: int
    ) -> list[Fill]:
        """Submit a limit order; return any immediate fills (may be empty).

        Marketable portion fills at touch against historical liquidity and the
        eaten volume is recorded as consumed (so this engine no longer sees it).
        Any remainder rests in the engine view.
        """
        snap = self._hist.latest
        fills: list[Fill] = []
        remaining = size
        opp: Side = "sell" if side == "buy" else "buy"
        levels = snap.asks if side == "buy" else snap.bids

        for px, hist_sz in levels:
            if remaining <= 1e-12:
                break
            crosses = px <= price if side == "buy" else px >= price
            if not crosses:
                break
            available = hist_sz - self._view.consumed_at(opp, px)
            if available <= 1e-12:
                continue
            take = min(remaining, available)
            # Fill at touch: execute at the resting historical price `px`.
            fills.append(
                Fill(
                    order_id=order_id,
                    ts=ts,
                    side=side,
                    price=px,
                    size=take,
                    mid_at_fill=snap.mid(),
                )
            )
            self._view.consume_historical(opp, px, take)
            remaining -= take

        if remaining > 1e-12:
            volume_ahead = max(
                0.0, snap.volume_at(side, price) - self._view.consumed_at(side, price)
            )
            self._view.add_own(
                OwnOrder(order_id, side, price, remaining, ts, volume_ahead)
            )
        return fills

    def check_resting_fills(self) -> list[Fill]:
        """Re-evaluate resting own orders against the latest book.

        Fill-at-touch: a resting buy fills once the best historical ask is at or
        below its price (the market traded down to it); symmetric for sells. The
        whole resting size fills at touch (simplest model; queue/`volume_ahead`
        refinement is a later step).
        """
        snap = self._hist.latest
        fills: list[Fill] = []
        for order in self._view.own_orders():
            opp: Side = "sell" if order.side == "buy" else "buy"
            best_opp = snap.best_ask() if order.side == "buy" else snap.best_bid()
            if math.isnan(best_opp):
                continue
            touched = (
                best_opp <= order.price
                if order.side == "buy"
                else best_opp >= order.price
            )
            if not touched:
                continue
            available = snap.volume_at(opp, best_opp) - self._view.consumed_at(
                opp, best_opp
            )
            if available <= 1e-12:
                continue
            take = min(order.size, available)
            fills.append(
                Fill(
                    order_id=order.order_id,
                    ts=snap.ts,
                    side=order.side,
                    price=order.price,  # fill at our resting (touch) price
                    size=take,
                    mid_at_fill=snap.mid(),
                )
            )
            self._view.consume_historical(opp, best_opp, take)
            self._view.reduce_own(order.order_id, take)
        return fills

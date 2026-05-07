"""Vanilla Avellaneda–Stoikov (2008) market-making strategy.

Reference price (with terminal-time penalty):
    r(s, q, t) = s − q · γ · σ² · (T − t)

Optimal total spread:
    δ_a + δ_b = γ · σ² · (T − t) + (2 / γ) · log(1 + γ / k)

Quotes are placed symmetrically around the reservation price r:
    p_bid = r − (δ_a + δ_b) / 2
    p_ask = r + (δ_a + δ_b) / 2

The "T − t" term collapses to a finite-horizon penalty; for an open-ended
backtest we use a rolling 1-hour horizon (config-driven). With
``T_horizon_seconds = ∞`` (encoded as a non-positive value) the inventory
penalty γ·σ²·(T−t) drops to a fixed γ·σ² per the infinite-horizon limit, so
the strategy still incentivises mean-reverting inventory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..lob.book import OrderBookState
from ..types import Event, QuoteAction
from .base import Strategy, StrategyState


@dataclass(slots=True)
class ASParams:
    gamma: float
    sigma: float
    k: float
    T_horizon_us: int  # rolling horizon in μs; ≤ 0 ⇒ infinite-horizon limit
    order_size: float
    max_inventory: float
    quote_refresh_min_interval_us: int = 100_000  # 100 ms default
    inventory_penalty_floor: float = 0.0


def _time_to_horizon(now_us: int, horizon_us: int) -> float:
    """Time remaining in seconds. With horizon_us ≤ 0 returns 1.0 (∞-horizon)."""
    if horizon_us <= 0:
        return 1.0
    elapsed = now_us % horizon_us
    return max(1e-6, (horizon_us - elapsed) / 1e6)  # in seconds


def reservation_price(s: float, q: float, gamma: float, sigma: float, dt: float) -> float:
    return s - q * gamma * sigma * sigma * dt


def optimal_spread(gamma: float, sigma: float, dt: float, k: float) -> float:
    return gamma * sigma * sigma * dt + (2.0 / gamma) * math.log1p(gamma / k)


class AvellanedaStoikovStrategy(Strategy):
    name = "as_vanilla"

    def __init__(self, params: ASParams) -> None:
        self.p = params

    def _reference_price(self, book: OrderBookState) -> float:
        return book.mid()

    def on_event(
        self,
        book: OrderBookState,
        event: Event,
        state: StrategyState,
    ) -> list[QuoteAction] | None:
        if not book.is_initialised():
            return None

        # Throttle quoting frequency to avoid cancelling on every micro-event.
        if state.last_quote_ts > 0 and event.ts - state.last_quote_ts < self.p.quote_refresh_min_interval_us:
            return None
        state.last_quote_ts = event.ts

        s = self._reference_price(book)
        q = state.inventory
        dt = _time_to_horizon(event.ts, self.p.T_horizon_us)

        r = reservation_price(s, q, self.p.gamma, self.p.sigma, dt)
        spread = optimal_spread(self.p.gamma, self.p.sigma, dt, self.p.k)
        half = 0.5 * spread

        bid_px = r - half
        ask_px = r + half

        # Prevent crossing: a buy quote must sit strictly below the best ask;
        # a sell quote must sit strictly above the best bid. We do not clamp
        # to (best bid, best ask) — improving on the touch is fine and is in
        # fact the regime where AS earns spread.
        if bid_px >= book.ask_px:
            bid_px = book.ask_px - max(book.spread() * 1e-3, 1e-12)
        if ask_px <= book.bid_px:
            ask_px = book.bid_px + max(book.spread() * 1e-3, 1e-12)

        actions: list[QuoteAction] = []
        if q < self.p.max_inventory:
            actions.append(QuoteAction(side="buy", price=bid_px, size=self.p.order_size))
        if q > -self.p.max_inventory:
            actions.append(QuoteAction(side="sell", price=ask_px, size=self.p.order_size))
        return actions

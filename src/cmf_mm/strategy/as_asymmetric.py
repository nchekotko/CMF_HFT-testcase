"""V3: Avellaneda–Stoikov + micro-price + imbalance-driven asymmetric skew.

Half-spreads are skewed by book imbalance I ∈ [−1, 1] to lean *against*
imbalance and reduce adverse selection (Cartea–Jaimungal-style):

    δ_a = δ* + α·I       (wider ask when I > 0, sell less into anticipated rise)
    δ_b = δ* − α·I       (tighter bid when I > 0, buy more before the rise)

Equivalently:
    bid_px = r − (δ* − α·I) = r − δ* + α·I
    ask_px = r + (δ* + α·I) = r + δ* + α·I

The constraint α < δ* keeps the strategy from crossing its own quotes; the
config validator enforces a sensible range.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lob.book import OrderBookState
from ..lob.microprice import MicropriceEstimator, WeightedMidEstimator
from ..types import Event, QuoteAction
from .avellaneda_stoikov import (
    ASParams,
    _time_to_horizon,
    optimal_spread,
    reservation_price,
)
from .base import Strategy, StrategyState


@dataclass(slots=True)
class ASAsymmetricParams(ASParams):
    alpha: float = 0.0  # skew coefficient on imbalance


class ASAsymmetricStrategy(Strategy):
    name = "as_asymmetric"

    def __init__(
        self,
        params: ASAsymmetricParams,
        microprice_estimator: MicropriceEstimator | None = None,
    ) -> None:
        self.p = params
        self._micro = microprice_estimator or WeightedMidEstimator()

    def on_event(
        self,
        book: OrderBookState,
        event: Event,
        state: StrategyState,
    ) -> list[QuoteAction] | None:
        if not book.is_initialised():
            return None
        if state.last_quote_ts > 0 and event.ts - state.last_quote_ts < self.p.quote_refresh_min_interval_us:
            return None
        state.last_quote_ts = event.ts

        s = self._micro.estimate(book)
        q = state.inventory
        dt = _time_to_horizon(event.ts, self.p.T_horizon_us)

        r = reservation_price(s, q, self.p.gamma, self.p.sigma, dt)
        spread = optimal_spread(self.p.gamma, self.p.sigma, dt, self.p.k)
        half = 0.5 * spread

        imb = book.imbalance()
        # lean *against* imbalance to reduce adverse selection:
        # I > 0 (bid heavy ⇒ price expected up): tighten bid, widen ask.
        bid_px = r - (half - self.p.alpha * imb)
        ask_px = r + (half + self.p.alpha * imb)

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

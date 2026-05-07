"""Top-of-book order-book state.

The engine operates on top-of-book quotes only. Deeper levels live in the
parquet file and can be loaded ad-hoc by analysis notebooks; the live event
stream stays narrow to keep per-event work tight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import LOBEvent


@dataclass(slots=True)
class OrderBookState:
    bid_px: float = math.nan
    bid_sz: float = 0.0
    ask_px: float = math.nan
    ask_sz: float = 0.0
    last_update_ts: int = 0

    def is_initialised(self) -> bool:
        return not (math.isnan(self.bid_px) or math.isnan(self.ask_px))

    def mid(self) -> float:
        return 0.5 * (self.bid_px + self.ask_px)

    def spread(self) -> float:
        return self.ask_px - self.bid_px

    def imbalance(self) -> float:
        """Order-flow imbalance:  I = (Q_b - Q_a) / (Q_b + Q_a) ∈ [-1, 1].

        Returns 0.0 when both sides are empty (degenerate state).
        """
        denom = self.bid_sz + self.ask_sz
        if denom <= 0.0:
            return 0.0
        return (self.bid_sz - self.ask_sz) / denom

    def update(self, event: LOBEvent) -> None:
        self.bid_px = event.bid_px
        self.bid_sz = event.bid_sz
        self.ask_px = event.ask_px
        self.ask_sz = event.ask_sz
        self.last_update_ts = event.ts

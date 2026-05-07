"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..lob.book import OrderBookState
from ..types import Event, QuoteAction


@dataclass(slots=True)
class StrategyState:
    inventory: float = 0.0
    cash: float = 0.0
    last_quote_ts: int = 0
    last_actions: list[QuoteAction] = field(default_factory=list)


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def on_event(
        self,
        book: OrderBookState,
        event: Event,
        state: StrategyState,
    ) -> list[QuoteAction] | None:
        """Return the desired quote state.

        Return values:
            None       — engine leaves active orders untouched (no-op tick).
            []         — engine cancels all active quotes.
            [...]      — engine reconciles to this set of quotes.
        """

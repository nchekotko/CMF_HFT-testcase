"""Micro-price estimators.

V2 of the strategy uses the volume-weighted top-of-book micro-price:

    p_micro = (Q_b · P_a + Q_a · P_b) / (Q_a + Q_b)

The interface is left as an ABC so a future iteration can plug in the full
Stoikov (2018) Markov-chain estimator without touching the strategy code.
That extension is sketched in ROADMAP.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .book import OrderBookState


class MicropriceEstimator(ABC):
    @abstractmethod
    def estimate(self, book: OrderBookState) -> float: ...


class WeightedMidEstimator(MicropriceEstimator):
    """Volume-weighted top-of-book micro-price.

    Equivalent to Gatheral–Oomen one-step micro-price under independent
    increments. Falls back to the arithmetic mid when one side has zero
    depth (degenerate state).
    """

    def estimate(self, book: OrderBookState) -> float:
        denom = book.bid_sz + book.ask_sz
        if denom <= 0.0:
            return book.mid()
        return (book.bid_sz * book.ask_px + book.ask_sz * book.bid_px) / denom


def weighted_mid(book: OrderBookState) -> float:
    return WeightedMidEstimator().estimate(book)


class StoikovMarkovChainEstimator(MicropriceEstimator):  # pragma: no cover - roadmap stub
    """Placeholder for the full Stoikov (2018) micro-price.

    Estimate G_∞ = R + Q · G_∞ on the (imbalance, spread) state space.
    Implemented in the roadmap; the stub is kept here so the strategy code
    can be wired against a stable interface today.
    """

    def estimate(self, book: OrderBookState) -> float:
        raise NotImplementedError("Roadmap item — see ROADMAP.md item 1.")

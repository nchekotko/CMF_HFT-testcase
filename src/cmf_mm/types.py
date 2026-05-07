"""Core dataclasses used across the backtester.

Timestamp convention: int microseconds since epoch (μs). The source CSVs use
microsecond resolution; we keep that throughout to avoid lossy or wasteful
unit conversion. Functions that need seconds derive them locally as `ts / 1e6`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NewType

OrderId = NewType("OrderId", int)

Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class TradeEvent:
    ts: int
    price: float
    size: float
    aggressor_side: Side


@dataclass(frozen=True, slots=True)
class LOBEvent:
    """Top-of-book snapshot. Higher levels are ignored by the engine but the
    raw L2 levels remain available in the parquet file for downstream analysis
    (e.g. queue-position extensions in the roadmap)."""

    ts: int
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float


Event = TradeEvent | LOBEvent


@dataclass(slots=True)
class Order:
    order_id: OrderId
    side: Side
    price: float
    size: float
    submit_ts: int


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: OrderId
    ts: int
    side: Side
    price: float
    size: float
    mid_at_fill: float


@dataclass(frozen=True, slots=True)
class QuoteAction:
    """Desired quote from a strategy. The engine diffs against active orders
    and issues cancel/place pairs as needed."""

    side: Side
    price: float
    size: float

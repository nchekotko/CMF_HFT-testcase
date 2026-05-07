"""Aggregated backtest summary."""

from __future__ import annotations

from dataclasses import dataclass

from .decomposition import PnLDecomposition
from .inventory import InventoryStats
from .pnl import PnLSeries


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    config_name: str
    pnl: PnLSeries
    inventory: InventoryStats
    decomposition: PnLDecomposition
    n_trades: int
    n_fills: int
    fill_rate: float
    turnover: float
    duration_seconds: float

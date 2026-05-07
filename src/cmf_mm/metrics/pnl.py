"""PnL series + headline metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PnLSeries:
    timestamps: np.ndarray  # μs
    realized_pnl: np.ndarray
    unrealized_pnl: np.ndarray
    total_pnl: np.ndarray

    def __post_init__(self) -> None:
        n = self.timestamps.shape[0]
        if not (
            self.realized_pnl.shape[0] == n
            and self.unrealized_pnl.shape[0] == n
            and self.total_pnl.shape[0] == n
        ):
            raise ValueError("PnLSeries arrays must share length")

    def sharpe(self, periods_per_year: float) -> float:
        if self.total_pnl.shape[0] < 2:
            return 0.0
        rets = np.diff(self.total_pnl)
        sd = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
        if sd == 0.0 or math.isnan(sd):
            return 0.0
        return float(rets.mean() / sd) * math.sqrt(periods_per_year)

    def max_drawdown(self) -> float:
        if self.total_pnl.size == 0:
            return 0.0
        running_max = np.maximum.accumulate(self.total_pnl)
        return float((self.total_pnl - running_max).min())

    def final(self) -> float:
        return float(self.total_pnl[-1]) if self.total_pnl.size else 0.0

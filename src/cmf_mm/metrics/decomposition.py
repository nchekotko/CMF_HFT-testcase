r"""P&L decomposition: spread capture vs inventory P&L.

Identity (continuous form, integrated over [0, T]):

    Delta PnL = Sum_{fills} (mid_t - fill_price) * (-signed_size)
                  ^ spread capture
              + integral of q_{t-} d mid_t
                  ^ inventory P&L

In discrete form, with mid sampled at every event,

    spread_capture += (fill_price - mid_at_fill) * size  for sell fills
                     (mid_at_fill - fill_price) * size  for buy fills
    inventory_pnl  += q_{t-} * (mid_t - mid_{t-1})

Total PnL:  spread_capture + inventory_pnl  ==  cash_t + q_t * mid_t

The class enforces the identity numerically as a class invariant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PnLDecomposition:
    spread_capture: float
    inventory_pnl: float
    total: float
    tol: float = 1e-6

    def __post_init__(self) -> None:
        residual = self.spread_capture + self.inventory_pnl - self.total
        if abs(residual) > self.tol:
            raise AssertionError(
                f"PnL decomposition identity violated: "
                f"spread_capture + inventory_pnl − total = {residual!r} (tol={self.tol})"
            )

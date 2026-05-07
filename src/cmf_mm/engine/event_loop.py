"""Event-driven backtest loop.

Loop structure:
  1. Pull next event.
  2. If LOBEvent: book.update(event); record (ts, mid, q) sample.
  3. If TradeEvent: feed it through the matcher against active orders;
     for each fill, update inventory, cash, and decomposition trackers.
  4. Ask the strategy for desired quotes; reconcile via OrderManager.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..lob.book import OrderBookState
from ..metrics.decomposition import PnLDecomposition
from ..metrics.inventory import inventory_stats
from ..metrics.pnl import PnLSeries
from ..metrics.summary import BacktestSummary
from ..strategy.base import Strategy, StrategyState
from ..types import Event, LOBEvent, TradeEvent
from .matcher import check_fills
from .order_manager import OrderManager


@dataclass(slots=True)
class BacktestConfig:
    config_name: str = "default"
    partial_fills: bool = False
    fee_bps: float = 0.0
    sample_every_n_events: int = 1  # downsample the recorded series


@dataclass(slots=True)
class BacktestResult:
    summary: BacktestSummary
    config_name: str


def run_backtest(
    events: Iterator[Event],
    strategy: Strategy,
    config: BacktestConfig,
) -> BacktestResult:
    book = OrderBookState()
    om = OrderManager()
    state = StrategyState()

    # Trackers
    cash = 0.0
    inventory = 0.0
    spread_capture = 0.0
    inventory_pnl = 0.0
    last_mid: float | None = None
    n_trades = 0
    n_fills = 0
    turnover = 0.0
    fee_rate = config.fee_bps / 1e4

    ts_list: list[int] = []
    mid_list: list[float] = []
    inv_list: list[float] = []
    cash_list: list[float] = []

    sample_counter = 0
    first_ts: int | None = None
    last_ts: int = 0

    for event in events:
        last_ts = event.ts
        if first_ts is None:
            first_ts = event.ts

        if isinstance(event, LOBEvent):
            book.update(event)
            new_mid = book.mid()
            if last_mid is not None:
                inventory_pnl += inventory * (new_mid - last_mid)
            last_mid = new_mid

        elif isinstance(event, TradeEvent):
            n_trades += 1
            if book.is_initialised():
                fills = check_fills(
                    event,
                    list(om.active_orders.values()),
                    mid_at_fill=book.mid(),
                    partial_fills=config.partial_fills,
                )
                for fill in fills:
                    n_fills += 1
                    om.remove(fill.order_id)
                    if fill.side == "buy":
                        inventory += fill.size
                        cash -= fill.price * fill.size
                        spread_capture += (fill.mid_at_fill - fill.price) * fill.size
                    else:
                        inventory -= fill.size
                        cash += fill.price * fill.size
                        spread_capture += (fill.price - fill.mid_at_fill) * fill.size
                    if fee_rate > 0.0:
                        fee = fee_rate * fill.price * fill.size
                        cash -= fee
                        spread_capture -= fee
                    turnover += fill.price * fill.size

        # Strategy step
        if book.is_initialised():
            state.inventory = inventory
            state.cash = cash
            actions = strategy.on_event(book, event, state)
            if actions is not None:
                om.reconcile(actions, ts=event.ts)
                state.last_actions = actions

        # Record
        if book.is_initialised():
            sample_counter += 1
            if sample_counter % config.sample_every_n_events == 0:
                ts_list.append(event.ts)
                mid_list.append(book.mid())
                inv_list.append(inventory)
                cash_list.append(cash)

    # Build PnL series
    if ts_list:
        ts_arr = np.asarray(ts_list, dtype=np.int64)
        mid_arr = np.asarray(mid_list, dtype=np.float64)
        inv_arr = np.asarray(inv_list, dtype=np.float64)
        cash_arr = np.asarray(cash_list, dtype=np.float64)
        total_arr = cash_arr + inv_arr * mid_arr
        # Realised vs unrealised: realised = cash flows; unrealised = inv·mid.
        realized = cash_arr
        unrealized = inv_arr * mid_arr
    else:
        ts_arr = np.zeros(0, dtype=np.int64)
        mid_arr = np.zeros(0, dtype=np.float64)
        inv_arr = np.zeros(0, dtype=np.float64)
        cash_arr = np.zeros(0, dtype=np.float64)
        realized = unrealized = total_arr = np.zeros(0, dtype=np.float64)

    pnl = PnLSeries(
        timestamps=ts_arr,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        total_pnl=total_arr,
    )
    inv = inventory_stats(ts_arr, inv_arr)

    total_pnl_final = float(total_arr[-1]) if total_arr.size else 0.0
    decomp = PnLDecomposition(
        spread_capture=spread_capture,
        inventory_pnl=inventory_pnl,
        total=total_pnl_final,
        tol=max(1e-6, abs(total_pnl_final) * 1e-9 + 1e-9),
    )

    duration_s = (last_ts - (first_ts or last_ts)) / 1e6
    fill_rate = n_fills / n_trades if n_trades > 0 else 0.0

    summary = BacktestSummary(
        config_name=config.config_name,
        pnl=pnl,
        inventory=inv,
        decomposition=decomp,
        n_trades=n_trades,
        n_fills=n_fills,
        fill_rate=fill_rate,
        turnover=turnover,
        duration_seconds=duration_s,
    )
    return BacktestResult(summary=summary, config_name=config.config_name)

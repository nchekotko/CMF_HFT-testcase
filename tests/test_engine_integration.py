"""End-to-end integration: a mini synthetic event stream through the loop."""

from __future__ import annotations

import math

from cmf_mm.engine.event_loop import BacktestConfig, run_backtest
from cmf_mm.strategy.avellaneda_stoikov import ASParams, AvellanedaStoikovStrategy
from cmf_mm.types import LOBEvent, TradeEvent


def _events():
    # Stable book around mid=100 with spread 1
    yield LOBEvent(ts=1_000, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0)
    # Trade that crosses our likely buy quote (sell-aggressor under mid)
    yield TradeEvent(ts=2_000, price=99.0, size=1.0, aggressor_side="sell")
    # Mid lifts
    yield LOBEvent(ts=3_000, bid_px=100.5, bid_sz=10.0, ask_px=101.5, ask_sz=10.0)
    # Trade that crosses our sell
    yield TradeEvent(ts=4_000, price=102.0, size=1.0, aggressor_side="buy")
    # End book
    yield LOBEvent(ts=5_000, bid_px=100.5, bid_sz=10.0, ask_px=101.5, ask_sz=10.0)


def test_engine_runs_end_to_end():
    p = ASParams(gamma=0.5, sigma=0.1, k=1.0, T_horizon_us=0,
                 order_size=1.0, max_inventory=10.0,
                 quote_refresh_min_interval_us=0)
    strat = AvellanedaStoikovStrategy(p)
    cfg = BacktestConfig(config_name="test", sample_every_n_events=1)
    res = run_backtest(_events(), strat, cfg)
    s = res.summary
    # The PnL decomposition class enforces the identity on construction.
    assert math.isclose(
        s.decomposition.spread_capture + s.decomposition.inventory_pnl,
        s.decomposition.total,
        abs_tol=1e-6,
    )
    assert s.n_trades == 2
    assert s.pnl.timestamps.size > 0

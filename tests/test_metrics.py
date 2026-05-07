from __future__ import annotations

import math

import numpy as np
import pytest

from cmf_mm.metrics.decomposition import PnLDecomposition
from cmf_mm.metrics.inventory import inventory_stats
from cmf_mm.metrics.pnl import PnLSeries


def test_decomposition_identity_holds():
    PnLDecomposition(spread_capture=2.0, inventory_pnl=3.0, total=5.0)


def test_decomposition_identity_violation_raises():
    with pytest.raises(AssertionError):
        PnLDecomposition(spread_capture=2.0, inventory_pnl=3.0, total=4.5)


def test_pnl_sharpe_is_zero_for_flat_series():
    n = 10
    s = PnLSeries(
        timestamps=np.arange(n, dtype=np.int64),
        realized_pnl=np.zeros(n),
        unrealized_pnl=np.zeros(n),
        total_pnl=np.zeros(n),
    )
    assert s.sharpe(periods_per_year=252) == 0.0


def test_pnl_max_drawdown_known_series():
    total = np.array([0.0, 1.0, 3.0, 2.0, -1.0, 0.0])
    s = PnLSeries(
        timestamps=np.arange(6, dtype=np.int64),
        realized_pnl=np.zeros(6),
        unrealized_pnl=total,
        total_pnl=total,
    )
    # peak at 3.0, trough at -1.0 ⇒ max_dd = -4.0
    assert math.isclose(s.max_drawdown(), -4.0)


def test_inventory_stats_basic():
    ts = np.array([0, 1, 2, 3], dtype=np.int64)
    inv = np.array([0.0, 1.0, 2.0, 1.0])
    stats = inventory_stats(ts, inv)
    assert stats.final == 1.0
    assert stats.max_abs == 2.0
    assert math.isclose(stats.mean, 1.0)


def test_pnl_decomposition_synthetic_full_path():
    """End-to-end identity on a hand-built sequence of fills + mid moves."""
    # initial state
    cash = 0.0
    inv = 0.0
    spread_capture = 0.0
    inventory_pnl = 0.0
    last_mid = 100.0

    # buy at 99.5 when mid=100  ⇒ spread_capture += 0.5
    fill_price, fill_size = 99.5, 1.0
    cash -= fill_price * fill_size
    inv += fill_size
    spread_capture += (last_mid - fill_price) * fill_size

    # mid moves to 101 with inventory=1  ⇒ inventory_pnl += 1.0
    new_mid = 101.0
    inventory_pnl += inv * (new_mid - last_mid)
    last_mid = new_mid

    # sell at 101.5 when mid=101  ⇒ spread_capture += 0.5
    fill_price, fill_size = 101.5, 1.0
    cash += fill_price * fill_size
    inv -= fill_size
    spread_capture += (fill_price - last_mid) * fill_size

    total = cash + inv * last_mid  # = -99.5 + 101.5 + 0 = 2.0
    PnLDecomposition(spread_capture=spread_capture, inventory_pnl=inventory_pnl, total=total)
    assert math.isclose(spread_capture + inventory_pnl, total, abs_tol=1e-9)
    assert math.isclose(spread_capture, 1.0)
    assert math.isclose(inventory_pnl, 1.0)
    assert math.isclose(total, 2.0)

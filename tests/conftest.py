"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cmf_mm.types import LOBEvent, Order, OrderId, TradeEvent


@pytest.fixture()
def tiny_book_event() -> LOBEvent:
    return LOBEvent(ts=1_000_000, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0)


@pytest.fixture()
def buy_order() -> Order:
    return Order(order_id=OrderId(1), side="buy", price=100.0, size=1.0, submit_ts=0)


@pytest.fixture()
def sell_order() -> Order:
    return Order(order_id=OrderId(2), side="sell", price=100.0, size=1.0, submit_ts=0)


@pytest.fixture()
def synthetic_intensity_data(tmp_path):
    """Generate a small synthetic dataset where δ has known A, k.

    We hold the mid constant, generate trade prices with δ ~ Exponential(k=2)
    on alternating sides. Total wall-clock time = N / A seconds.
    """
    rng = np.random.default_rng(42)
    n = 50_000
    # A is the *pooled* rate density at δ=0 (both sides combined). With
    # per-side rate λ_side = A_side·exp(-k·δ) and integral A_side/k = 5/sec,
    # we set A_side=10, k=2, total rate per side 5/sec, n trades over 5000s.
    # Pooled A (returned by the calibrator) = 2·A_side = 20.
    A_true = 20.0
    k_true = 2.0
    duration_s = n / 10.0  # 10 trades/sec total (5/sec per side)

    # spaced uniformly over the window
    ts_us = (np.linspace(0, duration_s, n, endpoint=False) * 1e6).astype(np.int64)
    delta = rng.exponential(scale=1.0 / k_true, size=n)
    side = rng.choice(["buy", "sell"], size=n)
    mid = 100.0
    price = np.where(side == "buy", mid + delta, mid - delta)

    trades_path = tmp_path / "trades.parquet"
    lob_path = tmp_path / "lob.parquet"

    pl.DataFrame({
        "ts": ts_us,
        "side": side,
        "price": price,
        "amount": np.full(n, 1.0),
    }).write_parquet(trades_path)

    pl.DataFrame({
        "ts": ts_us[::100],
        "ask_px_0": np.full(len(ts_us[::100]), mid + 0.01),
        "ask_sz_0": np.full(len(ts_us[::100]), 5.0),
        "bid_px_0": np.full(len(ts_us[::100]), mid - 0.01),
        "bid_sz_0": np.full(len(ts_us[::100]), 5.0),
    }).write_parquet(lob_path)

    return {
        "trades_path": str(trades_path),
        "lob_path": str(lob_path),
        "A_true": A_true,
        "k_true": k_true,
        "n": n,
    }


@pytest.fixture()
def trade_below() -> TradeEvent:
    return TradeEvent(ts=2_000_000, price=99.5, size=1.0, aggressor_side="sell")


@pytest.fixture()
def trade_above() -> TradeEvent:
    return TradeEvent(ts=2_000_000, price=100.5, size=1.0, aggressor_side="buy")

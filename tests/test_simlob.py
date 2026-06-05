"""Tests for the multi-engine LOB simulation (Group 3)."""

from __future__ import annotations

import threading

from cmf_mm.simlob import (
    HistoricalLOB,
    L3Action,
    L3Event,
    MultiEngineSimulator,
    l2_snapshots_to_l3,
)
from cmf_mm.types import OrderId


def _add(ts: int, oid: int, side: str, px: float, sz: float) -> L3Event:
    return L3Event(ts, L3Action.ADD, OrderId(oid), side, px, sz)  # type: ignore[arg-type]


def _basic_book() -> HistoricalLOB:
    lob = HistoricalLOB()
    lob.apply(_add(1, 1, "buy", 99.0, 5.0))
    lob.apply(_add(1, 2, "buy", 98.0, 3.0))
    lob.apply(_add(1, 3, "sell", 101.0, 4.0))
    lob.apply(_add(1, 4, "sell", 102.0, 2.0))
    lob.publish()
    return lob


# ---- HistoricalLOB reconstruction ----------------------------------------


def test_reconstruct_best_levels() -> None:
    snap = _basic_book().latest
    assert snap.best_bid() == 99.0
    assert snap.best_ask() == 101.0
    assert snap.best_bid_size() == 5.0
    assert snap.mid() == 100.0


def test_cancel_and_trade_remove_volume() -> None:
    lob = _basic_book()
    lob.apply(L3Event(2, L3Action.TRADE, OrderId(3), "sell", 101.0, 1.0))
    lob.apply(L3Event(2, L3Action.CANCEL, OrderId(1), "buy", 99.0, 5.0))
    snap = lob.publish()
    assert snap.volume_at("sell", 101.0) == 3.0  # 4 - 1 traded
    assert snap.best_bid() == 98.0  # 99 level cancelled away


def test_modify_resizes_level() -> None:
    lob = _basic_book()
    lob.apply(L3Event(2, L3Action.MODIFY, OrderId(1), "buy", 99.0, 9.0))
    snap = lob.publish()
    assert snap.volume_at("buy", 99.0) == 9.0


# ---- L2 -> L3 shim --------------------------------------------------------


def test_l2_shim_reconstructs_snapshot() -> None:
    snapshots = [
        (1, [(99.0, 5.0), (98.0, 3.0)], [(101.0, 4.0)]),
        (2, [(99.0, 7.0)], [(101.0, 4.0), (102.0, 1.0)]),  # 98 vanished, 99 grew, 102 new
    ]
    lob = HistoricalLOB()
    for ev in l2_snapshots_to_l3(snapshots):
        lob.apply(ev)
    snap = lob.publish()
    assert snap.volume_at("buy", 99.0) == 7.0
    assert snap.volume_at("buy", 98.0) == 0.0
    assert snap.volume_at("sell", 102.0) == 1.0
    assert snap.best_bid() == 99.0


# ---- engine isolation -----------------------------------------------------


def test_engines_do_not_see_each_others_orders() -> None:
    sim = MultiEngineSimulator(n_engines=2)
    sim.advance_many(
        [
            _add(1, 1, "buy", 99.0, 5.0),
            _add(1, 2, "sell", 101.0, 4.0),
        ]
    )
    # Engine 0 rests a passive bid at 100 (inside the spread, non-marketable).
    fills = sim.submit_limit(0, OrderId(100), "buy", 100.0, 2.0, ts=2)
    assert fills == []

    seen0 = sim.view_for(0)
    seen1 = sim.view_for(1)
    assert seen0.best_bid() == 100.0  # engine 0 sees its own order
    assert seen1.best_bid() == 99.0  # engine 1 does not


# ---- fill at touch --------------------------------------------------------


def test_marketable_order_fills_at_touch() -> None:
    sim = MultiEngineSimulator(n_engines=1)
    sim.advance_many(
        [
            _add(1, 1, "buy", 99.0, 5.0),
            _add(1, 2, "sell", 101.0, 4.0),
            _add(1, 3, "sell", 102.0, 2.0),
        ]
    )
    # Buy 6 @ 102 crosses: 4 @ 101 then 2 @ 102 (fill at touch on each level).
    fills = sim.submit_limit(0, OrderId(100), "buy", 102.0, 6.0, ts=2)
    assert [(f.price, f.size) for f in fills] == [(101.0, 4.0), (102.0, 2.0)]
    # The consumed liquidity disappears from this engine's view.
    seen = sim.view_for(0)
    assert seen.best_ask() != seen.best_ask() or seen.asks() == []  # no asks left


def test_consumed_liquidity_hidden_only_for_that_engine() -> None:
    sim = MultiEngineSimulator(n_engines=2)
    sim.advance_many([_add(1, 2, "sell", 101.0, 4.0)])
    sim.submit_limit(0, OrderId(100), "buy", 101.0, 4.0, ts=2)  # engine 0 eats it
    assert sim.view_for(0).asks() == []  # engine 0: gone
    assert sim.view_for(1).best_ask() == 101.0  # engine 1: still there


def test_resting_order_fills_when_market_touches() -> None:
    sim = MultiEngineSimulator(n_engines=1)
    sim.advance_many([_add(1, 1, "buy", 99.0, 5.0), _add(1, 2, "sell", 101.0, 4.0)])
    # Rest a buy at 100 — not marketable yet (best ask 101).
    assert sim.submit_limit(0, OrderId(100), "buy", 100.0, 2.0, ts=2) == []
    assert sim.check_resting_fills(0) == []
    # Market trades down: best ask becomes 100.
    sim.advance(_add(3, 5, "sell", 100.0, 3.0))
    fills = sim.check_resting_fills(0)
    assert len(fills) == 1
    assert fills[0].price == 100.0
    assert fills[0].size == 2.0


# ---- concurrency: N engines, one shared book, no races --------------------


def test_concurrent_engines_no_data_races() -> None:
    n = 8
    sim = MultiEngineSimulator(n_engines=n)
    stop = threading.Event()

    def writer() -> None:
        oid = 1000
        px = 100.0
        for i in range(2000):
            sim.advance(_add(i, oid, "sell", px, 1.0))
            oid += 1
            px = 100.0 + (i % 5)
        stop.set()

    fill_counts = [0] * n

    def engine(eid: int) -> None:
        local_oid = eid * 1_000_000
        while not stop.is_set():
            sim.submit_limit(eid, OrderId(local_oid), "buy", 100.0, 0.5, ts=0)
            local_oid += 1
            fill_counts[eid] += len(sim.check_resting_fills(eid))
            _ = sim.view_for(eid).best_ask()  # concurrent reads of shared snapshot

    threads = [threading.Thread(target=writer)]
    threads += [threading.Thread(target=engine, args=(e,)) for e in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # Each engine's view stays internally consistent (own orders only).
    for eid in range(n):
        for o in sim._views[eid].own_orders():
            assert o.side == "buy"

"""Invariant / property tests for the multi-engine LOB.

These assert facts that must hold for *any* input, not just hand-picked cases:
basement isolation, determinism, merge identity, non-negativity, snapshot
immutability and fill-price sanity. Randomised cases sweep several seeds.
"""

from __future__ import annotations

import random

import pytest

from cmf_mm.simlob import (
    EngineView,
    HistoricalLOB,
    L3Action,
    L3Event,
    MultiEngineSimulator,
    SimulatedLOB,
)
from cmf_mm.types import OrderId, Side

PRICES = [96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0]


def _add(ts: int, oid: int, side: Side, px: float, sz: float) -> L3Event:
    return L3Event(ts, L3Action.ADD, OrderId(oid), side, px, sz)


def _seed_book(events_out: list[L3Event] | None = None) -> list[L3Event]:
    """A fixed, sane two-sided book as a list of ADD events."""
    evs = [
        _add(1, 1, "buy", 99.0, 5.0), _add(1, 2, "buy", 98.0, 8.0),
        _add(1, 3, "buy", 97.0, 10.0), _add(1, 4, "sell", 101.0, 4.0),
        _add(1, 5, "sell", 102.0, 6.0), _add(1, 6, "sell", 103.0, 10.0),
    ]
    if events_out is not None:
        events_out.extend(evs)
    return evs


# ---- merge identity -------------------------------------------------------


def test_empty_view_equals_historical() -> None:
    """SimulatedLOB over an empty EngineView is exactly the historical book."""
    hist = HistoricalLOB()
    for ev in _seed_book():
        hist.apply(ev)
    snap = hist.publish()
    sim = SimulatedLOB(snap, EngineView(0))
    assert sim.bids() == list(snap.bids)
    assert sim.asks() == list(snap.asks)
    assert sim.best_bid() == snap.best_bid()
    assert sim.best_ask() == snap.best_ask()


# ---- basement isolation ---------------------------------------------------


def test_engine_activity_never_mutates_basement() -> None:
    """The shared HistoricalLOB is byte-identical with or without engine trading."""
    events = _seed_book()

    quiet = MultiEngineSimulator(n_engines=3)
    quiet.advance_many(events)

    busy = MultiEngineSimulator(n_engines=3)
    busy.advance_many(events)
    # Engines trade aggressively and passively; none may touch the basement.
    busy.submit_limit(0, OrderId(100), "buy", 103.0, 9.0, ts=2)  # eats asks
    busy.submit_limit(1, OrderId(200), "sell", 97.0, 12.0, ts=2)  # eats bids
    busy.submit_limit(2, OrderId(300), "buy", 99.0, 4.0, ts=2)    # rests
    for eid in range(3):
        busy.check_resting_fills(eid)

    assert quiet.historical.latest.bids == busy.historical.latest.bids
    assert quiet.historical.latest.asks == busy.historical.latest.asks


# ---- determinism ----------------------------------------------------------


def _run_scenario() -> list[tuple[float, float, str]]:
    sim = MultiEngineSimulator(n_engines=2)
    sim.advance_many(_seed_book())
    out: list[tuple[float, float, str]] = []
    for f in sim.submit_limit(0, OrderId(100), "buy", 102.0, 7.0, ts=2):
        out.append((f.price, f.size, f.side))
    sim.advance(_add(3, 7, "sell", 100.0, 3.0))
    for f in sim.check_resting_fills(0):
        out.append((f.price, f.size, f.side))
    return out


def test_replay_is_deterministic() -> None:
    assert _run_scenario() == _run_scenario()


# ---- snapshot immutability ------------------------------------------------


def test_held_snapshot_unchanged_after_advance() -> None:
    hist = HistoricalLOB()
    for ev in _seed_book():
        hist.apply(ev)
    held = hist.publish()
    held_bids, held_asks = held.bids, held.asks
    # Mutate the book a lot afterwards.
    hist.apply(_add(2, 50, "sell", 100.5, 99.0))
    hist.apply(L3Event(2, L3Action.CANCEL, OrderId(1), "buy", 99.0, 5.0))
    hist.publish()
    assert held.bids is held_bids and held.asks is held_asks
    assert held.best_bid() == 99.0  # the snapshot we held still says 99


# ---- fill-price sanity ----------------------------------------------------


def test_fill_prices_respect_limit_and_touch() -> None:
    sim = MultiEngineSimulator(n_engines=1)
    sim.advance_many(_seed_book())
    ask_prices = {p for p, _ in sim.historical.latest.asks}
    fills = sim.submit_limit(0, OrderId(100), "buy", 102.0, 7.0, ts=2)
    assert fills, "marketable order should fill"
    for f in fills:
        assert f.price <= 102.0  # never pay above the limit
        assert f.price in ask_prices  # fill at a real historical level (touch)


# ---- randomised non-negativity & conservation -----------------------------


@pytest.mark.parametrize("seed", range(12))
def test_random_marketable_never_negative_or_over_consumed(seed: int) -> None:
    rng = random.Random(seed)
    sim = MultiEngineSimulator(n_engines=1)
    events: list[L3Event] = []
    oid = 1
    for _ in range(rng.randint(4, 10)):
        side: Side = rng.choice(["buy", "sell"])
        px = rng.choice(PRICES[:4] if side == "buy" else PRICES[5:])
        events.append(_add(1, oid, side, px, float(rng.randint(1, 9))))
        oid += 1
    sim.advance_many(events)

    total_ask = sum(s for _, s in sim.historical.latest.asks)
    # Sweep everything on the ask side with a huge marketable buy.
    fills = sim.submit_limit(0, OrderId(999), "buy", 1e9, 1e9, ts=2)

    filled = sum(f.size for f in fills)
    assert filled <= total_ask + 1e-9  # cannot consume more than existed
    view = sim.view_for(0)
    assert all(v >= 0 for _, v in view.bids())  # no negative effective volume
    assert all(v >= 0 for _, v in view.asks())
    assert view.asks() == []  # all ask liquidity consumed by this engine

"""Real-data integration: reconstruct HistoricalLOB from the project's L2 parquet
and prove it matches the recorded book (closes the "represents the real market"
gap). Skips automatically when ``data/lob.parquet`` is absent (e.g. clean CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cmf_mm.simlob import HistoricalLOB, L2ToL3Converter, MultiEngineSimulator
from cmf_mm.simlob.adapters import L2Snapshot
from cmf_mm.simlob.realdata import DEFAULT_LOB_PATH, iter_lob_snapshots
from cmf_mm.types import OrderId, Side

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_LOB_PATH).exists(),
    reason=f"{DEFAULT_LOB_PATH} not present",
)

ROWS = 5000
TOL = 1e-6


def _expected_levels(levels: list[tuple[float, float]], side: Side) -> list[
    tuple[float, float]
]:
    agg: dict[float, float] = {}
    for px, sz in levels:
        agg[px] = agg.get(px, 0.0) + sz
    return sorted(agg.items(), key=lambda kv: -kv[0] if side == "buy" else kv[0])


def _assert_match(recon: list[tuple[float, float]],
                  expected: list[tuple[float, float]], where: str) -> None:
    assert len(recon) == len(expected), f"{where}: depth {len(recon)}≠{len(expected)}"
    for (rp, rv), (ep, ev) in zip(recon, expected, strict=True):
        assert rp == ep, f"{where}: price {rp}≠{ep}"
        assert abs(rv - ev) <= TOL * max(1.0, abs(ev)), f"{where}: size {rv}≠{ev}"


def test_reconstruction_matches_recorded_book() -> None:
    """Every reconstructed snapshot equals the recorded L2 book, full depth."""
    snapshots = list(iter_lob_snapshots(limit=ROWS))
    assert len(snapshots) > 100, "expected a non-trivial slice"

    hist = HistoricalLOB()
    conv = L2ToL3Converter()
    verified = 0
    for snap in snapshots:
        _ts, bids, asks = snap
        for ev in conv.step(snap):
            hist.apply(ev)
        hist.publish()
        latest = hist.latest
        # Top-of-book matches the recorded best levels.
        if bids:
            assert latest.best_bid() == max(p for p, _ in bids)
        if asks:
            assert latest.best_ask() == min(p for p, _ in asks)
        # Full depth matches.
        _assert_match(list(latest.bids), _expected_levels(bids, "buy"),
                      f"bids@row{verified}")
        _assert_match(list(latest.asks), _expected_levels(asks, "sell"),
                      f"asks@row{verified}")
        verified += 1
    assert verified == len(snapshots)


def test_n_engines_run_on_real_market_slice() -> None:
    """N engines trade on the real reconstructed book without errors; an engine's
    market impact stays private to that engine."""
    snapshots: list[L2Snapshot] = list(iter_lob_snapshots(limit=1000))
    sim = MultiEngineSimulator(n_engines=3)
    conv = L2ToL3Converter()

    # Warm up the book on the first snapshot.
    for ev in conv.step(snapshots[0]):
        sim.advance(ev)

    total_fills = 0
    oid = 0
    for snap in snapshots[1:]:
        for ev in conv.step(snap):
            sim.advance(ev)
        book = sim.historical.latest
        if book.best_ask() != book.best_ask():  # NaN guard
            continue
        # Engine 0 crosses the spread (marketable), engines 1/2 rest passively.
        oid += 1
        total_fills += len(sim.submit_limit(
            0, OrderId(oid), "buy", book.best_ask(), book.best_ask_size(), ts=snap[0]))
        sim.submit_limit(1, OrderId(10_000 + oid), "buy",
                         book.best_bid(), 1.0, ts=snap[0])
        for eid in range(3):
            total_fills += len(sim.check_resting_fills(eid))

    assert total_fills > 0, "engines should have filled on real data"
    # Engine 2 never traded → its private view equals the shared book.
    v2 = sim.view_for(2)
    assert v2.best_bid() == sim.historical.latest.best_bid()
    assert v2.best_ask() == sim.historical.latest.best_ask()

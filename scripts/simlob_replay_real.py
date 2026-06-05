"""Replay the real L2 parquet through the multi-engine LOB and report stats.

Demonstrates the production data path end-to-end on the actual HW1-HW2 dataset:

    data/lob.parquet  ->  L2 snapshots  ->  synthetic L3  ->  HistoricalLOB
                                                              + N EngineViews

It reconstructs the book, checks every reconstructed snapshot against the
recorded top-of-book, runs N engines (one aggressive taker, the rest passive),
and prints reconstruction accuracy, fills and a memory check (diffs vs copies).

Run:  uv run python scripts/simlob_replay_real.py --rows 20000 --engines 3
"""

from __future__ import annotations

import argparse
import sys

from cmf_mm.simlob import L2ToL3Converter, MultiEngineSimulator
from cmf_mm.simlob.realdata import iter_lob_snapshots
from cmf_mm.types import OrderId

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000, help="L2 rows to scan")
    ap.add_argument("--engines", type=int, default=3)
    args = ap.parse_args()

    sim = MultiEngineSimulator(n_engines=args.engines)
    conv = L2ToL3Converter()

    snaps = 0
    mismatches = 0
    fills = 0
    oid = 0
    quote_id: dict[int, OrderId] = {}  # each passive engine's live quote
    for snap in iter_lob_snapshots(limit=args.rows):
        ts, bids, asks = snap
        for ev in conv.step(snap):
            sim.advance(ev)
        book = sim.historical.latest
        snaps += 1

        # Reconstruction check against recorded top-of-book.
        if bids and book.best_bid() != max(p for p, _ in bids):
            mismatches += 1
        if asks and book.best_ask() != min(p for p, _ in asks):
            mismatches += 1

        if book.best_ask() != book.best_ask():  # NaN guard
            continue
        # Engine 0 lifts the offer (pure marketable taker): take the fills and
        # cancel any unfilled remainder so it never rests (a market order).
        oid += 1
        fills += len(sim.submit_limit(
            0, OrderId(oid), "buy", book.best_ask(), book.best_ask_size(), ts=ts))
        sim.cancel(0, OrderId(oid))
        # Engines 1..N re-quote at the bid: cancel the old quote, place a fresh
        # one — so their diff stays O(1), like a real market maker.
        for eid in range(1, args.engines):
            if eid in quote_id:
                sim.cancel(eid, quote_id[eid])
            qid = OrderId(eid * 10**7 + oid)
            quote_id[eid] = qid
            sim.submit_limit(eid, qid, "buy", book.best_bid(), 1.0, ts=ts)
        for eid in range(args.engines):
            got = sim.check_resting_fills(eid)
            fills += len(got)
            if got and eid in quote_id:  # quote filled → it is no longer live
                quote_id.pop(eid, None)

    book = sim.historical.latest
    print(f"snapshots reconstructed : {snaps:,}")
    print(f"top-of-book mismatches  : {mismatches}  "
          f"({'PERFECT' if mismatches == 0 else 'CHECK'})")
    print(f"final best bid / ask    : {book.best_bid():.8g} / {book.best_ask():.8g}")
    print(f"historical depth (levels): {len(book.bids)} bid / {len(book.asks)} ask")
    print(f"synthetic fills (all eng): {fills:,}")

    # Memory argument: the basement (O(depth) levels) is shared by all engines;
    # each engine stores only its live diff. With cancel/replace the own-order
    # set stays O(active quotes). 'consumed' accumulates because the synthetic
    # taker never has its impact expire — in a real run it would be reset/decayed
    # per replay window; here it is the cumulative footprint over the whole slice.
    depth = len(book.bids) + len(book.asks)
    own_live = sum(len(sim._views[e].own_orders()) for e in range(args.engines))
    consumed = sum(len(sim._views[e].consumed_levels()) for e in range(args.engines))
    print(f"shared basement levels  : {depth} (one copy, all engines)")
    print(f"live own orders (diff)  : {own_live}  "
          f"[N full copies would store {args.engines}×{depth} = {args.engines * depth} "
          "level-cells, duplicated every snapshot]")
    print(f"cumulative consumed keys: {consumed} (whole-slice footprint, not steady-state)")


if __name__ == "__main__":
    main()

"""Differential test: diff-based system vs a brute-force "N full copies" oracle.

The assignment's key design question is "store N full copies of the LOB, or store
diffs?". We chose diffs. This test proves the diff implementation is *equivalent*
to the rejected full-copy design by running both over randomised scenarios and
asserting identical books and fills at every step.

``OracleSim`` is a deliberately dumb, independent reimplementation: it keeps its
own dict-based historical book and, per engine, materialises the full effective
book on demand (the "N copies" approach) with an independently written
fill-at-touch routine. No code is shared with ``cmf_mm.simlob`` beyond the public
event/order types, so agreement between the two is real evidence of correctness.
"""

from __future__ import annotations

import math
import random

import pytest

from cmf_mm.simlob import L3Action, L3Event, MultiEngineSimulator
from cmf_mm.types import OrderId, Side

Levels = list[tuple[float, float]]


# --------------------------------------------------------------------------
# Independent oracle (full-copy design, plain dicts).
# --------------------------------------------------------------------------
class OracleHist:
    """Standalone historical reconstruction — mirrors HistoricalLOB semantics."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.orders: dict[int, tuple[Side, float, float]] = {}

    def _book(self, side: Side) -> dict[float, float]:
        return self.bids if side == "buy" else self.asks

    def apply(self, ev: L3Event) -> None:
        if ev.action is L3Action.ADD:
            assert ev.side is not None
            self.orders[int(ev.order_id)] = (ev.side, ev.price, ev.size)
            book = self._book(ev.side)
            book[ev.price] = book.get(ev.price, 0.0) + ev.size
        elif ev.action in (L3Action.TRADE, L3Action.CANCEL):
            o = self.orders.get(int(ev.order_id))
            if o is None:
                return
            oside, opx, osz = o
            qty = osz if ev.action is L3Action.CANCEL else min(ev.size, osz)
            book = self._book(oside)
            left = book.get(opx, 0.0) - qty
            if left <= 1e-12:
                book.pop(opx, None)
            else:
                book[opx] = left
            if osz - qty <= 1e-12:
                self.orders.pop(int(ev.order_id), None)
            else:
                self.orders[int(ev.order_id)] = (oside, opx, osz - qty)

    def levels(self, side: Side) -> Levels:
        book = self.bids if side == "buy" else self.asks
        return sorted(book.items(), key=lambda kv: -kv[0] if side == "buy" else kv[0])

    def best(self, side: Side) -> float:
        lv = self.levels(side)
        return lv[0][0] if lv else math.nan

    def volume_at(self, side: Side, px: float) -> float:
        return (self.bids if side == "buy" else self.asks).get(px, 0.0)


class OracleSim:
    def __init__(self, n: int) -> None:
        self.hist = OracleHist()
        self.consumed: list[dict[tuple[Side, float], float]] = [{} for _ in range(n)]
        self.own: list[dict[int, tuple[Side, float, float]]] = [{} for _ in range(n)]

    def advance(self, ev: L3Event) -> None:
        self.hist.apply(ev)

    def _consumed_at(self, eid: int, side: Side, px: float) -> float:
        return self.consumed[eid].get((side, px), 0.0)

    def effective(self, eid: int, side: Side) -> Levels:
        prices: dict[float, float] = {}
        for px, hv in self.hist.levels(side):
            eff = hv - self._consumed_at(eid, side, px)
            if eff > 1e-12:
                prices[px] = eff
        for oside, opx, osz in self.own[eid].values():
            if oside == side:
                prices[opx] = prices.get(opx, 0.0) + osz
        return sorted(prices.items(), key=lambda kv: -kv[0] if side == "buy" else kv[0])

    def submit_limit(
        self, eid: int, oid: int, side: Side, price: float, size: float
    ) -> list[tuple[float, float, str]]:
        opp: Side = "sell" if side == "buy" else "buy"
        levels = self.hist.levels(opp)
        remaining = size
        fills: list[tuple[float, float, str]] = []
        for px, hv in levels:
            crosses = px <= price if side == "buy" else px >= price
            if not crosses or remaining <= 1e-12:
                break
            avail = hv - self._consumed_at(eid, opp, px)
            if avail <= 1e-12:
                continue
            take = min(remaining, avail)
            fills.append((px, take, side))
            self.consumed[eid][(opp, px)] = self._consumed_at(eid, opp, px) + take
            remaining -= take
        if remaining > 1e-12:
            self.own[eid][oid] = (side, price, remaining)
        return fills

    def check_resting_fills(self, eid: int) -> list[tuple[float, float, str]]:
        fills: list[tuple[float, float, str]] = []
        for oid, (side, opx, osz) in list(self.own[eid].items()):
            opp: Side = "sell" if side == "buy" else "buy"
            best_opp = self.hist.best(opp)
            if math.isnan(best_opp):
                continue
            touched = best_opp <= opx if side == "buy" else best_opp >= opx
            if not touched:
                continue
            avail = self.hist.volume_at(opp, best_opp) - self._consumed_at(eid, opp, best_opp)
            if avail <= 1e-12:
                continue
            take = min(osz, avail)
            fills.append((opx, take, side))
            self.consumed[eid][(opp, best_opp)] = (
                self._consumed_at(eid, opp, best_opp) + take
            )
            if osz - take <= 1e-12:
                self.own[eid].pop(oid)
            else:
                self.own[eid][oid] = (side, opx, osz - take)
        return fills


# --------------------------------------------------------------------------
# Differential driver.
# --------------------------------------------------------------------------
def _round_levels(levels: Levels) -> list[tuple[float, float]]:
    return [(round(p, 6), round(s, 6)) for p, s in levels]


def _round_fills(
    fills: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    return [(round(p, 6), round(s, 6), side) for p, s, side in fills]


@pytest.mark.parametrize("seed", range(25))
def test_diffs_match_full_copies(seed: int) -> None:
    rng = random.Random(seed)
    n = 3
    sim = MultiEngineSimulator(n_engines=n)
    orc = OracleSim(n)
    bid_px = [96.0, 97.0, 98.0, 99.0, 100.0]
    ask_px = [100.0, 101.0, 102.0, 103.0, 104.0]
    oid = 0
    live: list[int] = []  # historical order ids still potentially in the book

    def hist_event() -> L3Event:
        nonlocal oid
        if live and rng.random() < 0.35:
            target = rng.choice(live)
            return L3Event(1, L3Action.TRADE, OrderId(target), None, math.nan,
                           float(rng.randint(1, 4)))
        oid += 1
        live.append(oid)
        side: Side = rng.choice(["buy", "sell"])
        px = rng.choice(bid_px if side == "buy" else ask_px)
        return L3Event(1, L3Action.ADD, OrderId(oid), side, px,
                       float(rng.randint(1, 9)))

    # Seed an initial book.
    for _ in range(rng.randint(4, 8)):
        ev = hist_event()
        sim.advance(ev)
        orc.advance(ev)

    eng_oid = 10_000
    for _step in range(60):
        roll = rng.random()
        if roll < 0.4:  # historical event
            ev = hist_event()
            sim.advance(ev)
            orc.advance(ev)
        elif roll < 0.75:  # engine submits a limit order
            eid = rng.randrange(n)
            side = rng.choice(["buy", "sell"])
            px = rng.choice(bid_px + ask_px)
            size = float(rng.randint(1, 6))
            eng_oid += 1
            a = sim.submit_limit(eid, OrderId(eng_oid), side, px, size, ts=2)
            b = orc.submit_limit(eid, eng_oid, side, px, size)
            assert _round_fills([(f.price, f.size, f.side) for f in a]) == \
                _round_fills(b), f"submit fills differ seed={seed} step={_step}"
        else:  # engine re-checks resting orders
            eid = rng.randrange(n)
            a = sim.check_resting_fills(eid)
            b = orc.check_resting_fills(eid)
            assert _round_fills([(f.price, f.size, f.side) for f in a]) == \
                _round_fills(b), f"resting fills differ seed={seed} step={_step}"

        # Books must agree everywhere, every step.
        assert _round_levels(list(sim.historical.latest.bids)) == \
            _round_levels(orc.hist.levels("buy")), f"hist bids seed={seed}"
        assert _round_levels(list(sim.historical.latest.asks)) == \
            _round_levels(orc.hist.levels("sell")), f"hist asks seed={seed}"
        for eid in range(n):
            view = sim.view_for(eid)
            assert _round_levels(view.bids()) == _round_levels(orc.effective(eid, "buy")), \
                f"engine {eid} bids differ seed={seed} step={_step}"
            assert _round_levels(view.asks()) == _round_levels(orc.effective(eid, "sell")), \
                f"engine {eid} asks differ seed={seed} step={_step}"

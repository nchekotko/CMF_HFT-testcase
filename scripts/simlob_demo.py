"""Demonstrate and visualise the multi-engine LOB simulation (Group 3).

Runs a scripted scenario through :class:`MultiEngineSimulator` and produces:

* a step-by-step **text trace** (stdout) so the logic is verifiable by eye;
* **depth-chart figures** — one panel per (stage x participant) showing the
  shared HistoricalLOB and each engine's *private* SimulatedLOB. Ghost bars are
  the real market; solid bars are what the engine actually sees; bold-edged bars
  are the engine's own injected orders. The gap between ghost and solid is the
  liquidity that engine already consumed.
* a **fills timeline** marking every synthetic fill per engine.

Run:  uv run python scripts/simlob_demo.py
Out:  results/simlob/depth_stages.png, results/simlob/fills_timeline.png
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from cmf_mm.simlob import L3Action, L3Event, MultiEngineSimulator
from cmf_mm.simlob.historical import BookSnapshot
from cmf_mm.types import Fill, OrderId, Side

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # UTF-8 stdout on Windows

OUT_DIR = Path("results/simlob")
ENGINE_NAMES = ["E0 passive-maker", "E1 aggressive-taker", "E2 queue-joiner"]
BID_C, ASK_C = "#2e8b57", "#c0392b"


def add(ts: int, oid: int, side: Side, px: float, sz: float) -> L3Event:
    return L3Event(ts, L3Action.ADD, OrderId(oid), side, px, sz)


@dataclass(slots=True)
class Capture:
    """A frozen record of the whole system at one stage, for plotting."""

    title: str
    hist: BookSnapshot
    # per engine: (bids, asks, own_prices_by_side, consumed_by_side)
    eng_bids: list[list[tuple[float, float]]] = field(default_factory=list)
    eng_asks: list[list[tuple[float, float]]] = field(default_factory=list)
    own: list[dict[Side, set[float]]] = field(default_factory=list)


def snapshot_engine(sim: MultiEngineSimulator, eid: int) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]], dict[Side, set[float]]
]:
    view = sim.view_for(eid)
    ev = sim._views[eid]  # demo introspection of private per-engine state
    own: dict[Side, set[float]] = {"buy": set(), "sell": set()}
    for o in ev.own_orders():
        own[o.side].add(o.price)
    return view.bids(), view.asks(), own


def capture(sim: MultiEngineSimulator, n_eng: int, title: str) -> Capture:
    cap = Capture(title=title, hist=sim.historical.latest)
    for eid in range(n_eng):
        b, a, own = snapshot_engine(sim, eid)
        cap.eng_bids.append(b)
        cap.eng_asks.append(a)
        cap.own.append(own)
    return cap


def trace_fills(label: str, fills: list[Fill]) -> None:
    if not fills:
        print(f"    {label}: (no fill — rests passively)")
        return
    for f in fills:
        print(
            f"    {label}: FILL {f.side} {f.size:g} @ {f.price:g} "
            f"(mid={f.mid_at_fill:g})"
        )


def run_scenario() -> tuple[list[Capture], list[tuple[int, str, Fill]]]:
    """Deterministic scripted scenario; returns plot captures + a fills log."""
    n_eng = 3
    sim = MultiEngineSimulator(n_engines=n_eng)
    fills_log: list[tuple[int, str, Fill]] = []
    caps: list[Capture] = []

    def log(stage_ts: int, eid: int, fills: list[Fill]) -> None:
        for f in fills:
            fills_log.append((stage_ts, ENGINE_NAMES[eid], f))

    # --- Stage 0: historical book built from L3 replay -------------------
    print("=" * 68)
    print("STAGE 0 — HistoricalLOB reconstructed from L3 replay (the basement)")
    print("=" * 68)
    sim.advance_many(
        [
            add(1, 1, "buy", 99.0, 5.0),
            add(1, 2, "buy", 98.0, 8.0),
            add(1, 3, "buy", 97.0, 10.0),
            add(1, 4, "sell", 101.0, 4.0),
            add(1, 5, "sell", 102.0, 6.0),
            add(1, 6, "sell", 103.0, 10.0),
        ]
    )
    h = sim.historical.latest
    print(f"  best bid={h.best_bid():g} ({h.best_bid_size():g})  "
          f"best ask={h.best_ask():g} ({h.best_ask_size():g})  mid={h.mid():g}")
    caps.append(capture(sim, n_eng, "Stage 0: initial book (all engines = market)"))

    # --- Stage 1: engines act -------------------------------------------
    print("\n" + "=" * 68)
    print("STAGE 1 — each engine injects its own orders")
    print("=" * 68)
    print("  E0 rests BUY 3 @ 100  (inside spread, non-marketable)")
    f0 = sim.submit_limit(0, OrderId(100), "buy", 100.0, 3.0, ts=2)
    trace_fills(ENGINE_NAMES[0], f0)
    log(2, 0, f0)

    print("  E1 sends BUY 7 @ 102 (marketable — should eat 101 then 102)")
    f1 = sim.submit_limit(1, OrderId(200), "buy", 102.0, 7.0, ts=2)
    trace_fills(ENGINE_NAMES[1], f1)
    log(2, 1, f1)

    print("  E2 rests BUY 4 @ 99  (joins existing 99 level behind the queue)")
    f2 = sim.submit_limit(2, OrderId(300), "buy", 99.0, 4.0, ts=2)
    trace_fills(ENGINE_NAMES[2], f2)
    log(2, 2, f2)

    print("\n  Private best-of-book AFTER actions (note divergence):")
    for eid in range(n_eng):
        v = sim.view_for(eid)
        print(f"    {ENGINE_NAMES[eid]:22s}  bid={v.best_bid():g}  ask={v.best_ask():g}")
    print(f"    {'HISTORICAL (shared)':22s}  bid={h.best_bid():g}  ask={h.best_ask():g}"
          "   <- unchanged for everyone else")
    caps.append(capture(sim, n_eng, "Stage 1: after each engine acts"))

    # --- Stage 2: market advances, resting orders re-checked -------------
    print("\n" + "=" * 68)
    print("STAGE 2 — market trades down: new historical SELL appears at 100")
    print("=" * 68)
    sim.advance(add(3, 7, "sell", 100.0, 3.0))
    print("  best ask is now 100 -> E0's resting BUY @ 100 should fill at touch")
    for eid in range(n_eng):
        rf = sim.check_resting_fills(eid)
        trace_fills(ENGINE_NAMES[eid], rf)
        log(3, eid, rf)
    caps.append(capture(sim, n_eng, "Stage 2: market touches E0's resting bid"))

    return caps, fills_log


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def _draw_book(
    ax: Axes,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    *,
    ghost_bids: list[tuple[float, float]] | None = None,
    ghost_asks: list[tuple[float, float]] | None = None,
    own: dict[Side, set[float]] | None = None,
    title: str = "",
) -> None:
    own = own or {"buy": set(), "sell": set()}
    # Ghost = real market (what historical has); drawn faint behind.
    for levels, color in ((ghost_bids, BID_C), (ghost_asks, ASK_C)):
        for px, sz in levels or []:
            ax.bar(px, sz, width=0.8, color=color, alpha=0.15, zorder=1)
    # Solid = what this participant actually sees.
    for levels, color, side in ((bids, BID_C, "buy"), (asks, ASK_C, "sell")):
        for px, sz in levels:
            is_own = px in own[side]  # type: ignore[index]
            ax.bar(
                px, sz, width=0.8, color=color, alpha=0.85, zorder=2,
                edgecolor="black" if is_own else "none",
                linewidth=2.2 if is_own else 0.0,
                hatch="///" if is_own else None,
            )
    all_px = [p for p, _ in bids + asks + (ghost_bids or []) + (ghost_asks or [])]
    if all_px:
        ax.set_xlim(min(all_px) - 1, max(all_px) + 1)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=8)


def plot_depth_stages(caps: list[Capture], out: Path) -> None:
    n_stage, n_eng = len(caps), len(caps[0].eng_bids)
    ncol = 1 + n_eng  # historical + engines
    fig, axes = plt.subplots(
        n_stage, ncol, figsize=(4.2 * ncol, 3.2 * n_stage), dpi=100, squeeze=False
    )
    for r, cap in enumerate(caps):
        hb = list(cap.hist.bids)
        ha = list(cap.hist.asks)
        _draw_book(axes[r][0], hb, ha, title=f"HISTORICAL\n{cap.title}")
        for e in range(n_eng):
            _draw_book(
                axes[r][e + 1],
                cap.eng_bids[e],
                cap.eng_asks[e],
                ghost_bids=hb,
                ghost_asks=ha,
                own=cap.own[e],
                title=ENGINE_NAMES[e],
            )
    # Legend
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=BID_C, alpha=0.85, label="bid (seen)"),
        Patch(facecolor=ASK_C, alpha=0.85, label="ask (seen)"),
        Patch(facecolor="grey", alpha=0.2, label="ghost = real market"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="own order"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Multi-engine LOB — each engine's private view vs the shared book",
                 y=1.02, fontsize=12)
    fig.supxlabel("price")
    fig.supylabel("size")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")


def plot_fills_timeline(fills_log: list[tuple[int, str, Fill]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
    names = ENGINE_NAMES
    ypos = {n: i for i, n in enumerate(names)}
    markers = {"buy": "^", "sell": "v"}
    # Spread fills that share the same (ts, engine) so markers/labels don't stack.
    seen: dict[tuple[int, str], int] = {}
    for ts, name, f in fills_log:
        k = (ts, name)
        idx = seen.get(k, 0)
        seen[k] = idx + 1
        x = ts + 0.10 * idx
        y = ypos[name]
        ax.scatter(
            x, y, s=140 + 30 * f.size, marker=markers[f.side],
            color=BID_C if f.side == "buy" else ASK_C, zorder=3, edgecolor="black",
        )
        ax.annotate(f"{f.side} {f.size:g}@{f.price:g}", (x, y),
                    textcoords="offset points", xytext=(8, 8 + 12 * idx),
                    fontsize=8)
    ax.margins(x=0.15)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(names)
    ax.set_xlabel("stage timestamp (μs)")
    ax.set_title("Synthetic fills per engine (▲ buy, ▼ sell; size ∝ qty)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_ylim(-0.5, len(names) - 0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    caps, fills_log = run_scenario()
    print("\n" + "=" * 68)
    print(f"SUMMARY — {len(fills_log)} synthetic fills generated")
    print("=" * 68)
    plot_depth_stages(caps, OUT_DIR / "depth_stages.png")
    plot_fills_timeline(fills_log, OUT_DIR / "fills_timeline.png")


if __name__ == "__main__":
    main()

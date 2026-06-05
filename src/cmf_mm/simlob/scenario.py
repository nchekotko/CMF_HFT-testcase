"""Scripted demo scenario for the multi-engine LOB — pure data, no plotting.

Drives :class:`MultiEngineSimulator` through a fixed sequence of L3-replay events
and engine decisions, capturing a :class:`Frame` after every step. Both the
matplotlib HTML player (``scripts/simlob_animate.py``) and the Streamlit
dashboard (``scripts/simlob_app.py``) consume these frames, so the two views are
always telling the exact same story.

A :class:`Frame` is a complete, frozen description of the whole system at one
step: the shared historical book, each engine's private book, which orders are
the engine's own, any fills generated, plus a human-readable decision log and a
tag for which architecture component is active (for the data-flow diagram).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Fill, OrderId, Side
from .events import L3Action, L3Event
from .historical import BookSnapshot
from .multi_engine import MultiEngineSimulator

N_ENG = 3
ENGINE_NAMES = ["E0 · passive-maker", "E1 · aggressive-taker", "E2 · queue-joiner"]


@dataclass(slots=True)
class EngineFrame:
    """One engine's private view at a step, for rendering."""

    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    own: dict[Side, set[float]]
    fills: list[Fill]


@dataclass(slots=True)
class Frame:
    """Complete system state at one animation step."""

    step: int
    ts: int
    active: str  # "replay" | f"engine{k}"
    eng_idx: int | None
    title: str
    log: list[str]
    hist: BookSnapshot
    engines: list[EngineFrame] = field(default_factory=list)

    def fills_this_step(self) -> list[tuple[int, Fill]]:
        """``(engine_idx, fill)`` pairs generated on this step."""
        out: list[tuple[int, Fill]] = []
        for k, ef in enumerate(self.engines):
            out.extend((k, f) for f in ef.fills)
        return out


def fmt_px(x: float) -> str:
    """Format a price, rendering NaN (empty side) as a dash."""
    return "—" if x != x else f"{x:g}"


# Architecture-diagram node keys, in pipeline order.
FLOW_NODES = ["source", "historical", "snapshot", "engine0", "engine1", "engine2", "fills"]
FLOW_EDGES = [
    ("source", "historical"), ("historical", "snapshot"),
    ("snapshot", "engine0"), ("snapshot", "engine1"), ("snapshot", "engine2"),
    ("engine0", "fills"), ("engine1", "fills"), ("engine2", "fills"),
]


def active_flow(frame: Frame) -> tuple[set[str], set[tuple[str, str]]]:
    """Which architecture nodes/edges are active (lit up) on this frame."""
    if frame.active == "replay":
        return ({"source", "historical", "snapshot"},
                {("source", "historical"), ("historical", "snapshot")})
    if frame.active.startswith("engine") and frame.eng_idx is not None:
        k = frame.eng_idx
        nodes = {"snapshot", f"engine{k}"}
        edges = {("snapshot", f"engine{k}")}
        if frame.engines[k].fills:
            nodes.add("fills")
            edges.add((f"engine{k}", "fills"))
        return nodes, edges
    return set(), set()


def _engine_frame(
    sim: MultiEngineSimulator, eid: int, fills: list[Fill]
) -> EngineFrame:
    view = sim.view_for(eid)
    ev = sim._views[eid]  # introspection for visualisation
    own: dict[Side, set[float]] = {"buy": set(), "sell": set()}
    for o in ev.own_orders():
        own[o.side].add(o.price)
    return EngineFrame(view.bids(), view.asks(), own, fills)


def _capture(
    sim: MultiEngineSimulator,
    step: int,
    active: str,
    eng_idx: int | None,
    title: str,
    log: list[str],
    fills_by_engine: dict[int, list[Fill]] | None = None,
) -> Frame:
    fills_by_engine = fills_by_engine or {}
    h = sim.historical.latest
    fr = Frame(step, h.ts, active, eng_idx, title, log, h)
    for eid in range(N_ENG):
        fr.engines.append(_engine_frame(sim, eid, fills_by_engine.get(eid, [])))
    return fr


def build_frames() -> list[Frame]:
    """Run the scripted scenario and return one frame per step."""
    sim = MultiEngineSimulator(n_engines=N_ENG)
    frames: list[Frame] = []
    step = 0

    def emit(
        active: str,
        eng: int | None,
        title: str,
        log: list[str],
        fills: dict[int, list[Fill]] | None = None,
    ) -> None:
        nonlocal step
        frames.append(_capture(sim, step, active, eng, title, log, fills))
        step += 1

    # Phase A — reconstruct HistoricalLOB from an L3 replay, one event per frame.
    book_events: list[tuple[Side, float, float]] = [
        ("buy", 99.0, 5.0), ("buy", 98.0, 8.0), ("buy", 97.0, 10.0),
        ("sell", 101.0, 4.0), ("sell", 102.0, 6.0), ("sell", 103.0, 10.0),
    ]
    oid = 1
    for side, px, sz in book_events:
        sim.advance(L3Event(1, L3Action.ADD, OrderId(oid), side, px, sz))
        h = sim.historical.latest
        emit(
            "replay", None,
            f"L3 replay — ADD {side} {sz:g} @ {px:g}",
            [
                "Data Source emits an L3 event (Databento MBO / L2-shim).",
                "HistoricalLOB.apply() folds it into the real book.",
                "publish() exposes a fresh immutable BookSnapshot to all engines.",
                f"best bid {fmt_px(h.best_bid())}  |  best ask {fmt_px(h.best_ask())}",
            ],
        )
        oid += 1

    # Phase B — each engine injects its own orders (the decisions).
    f = sim.submit_limit(0, OrderId(100), "buy", 100.0, 3.0, ts=2)
    emit(
        "engine0", 0, "E0 decision — rest BUY 3 @ 100 (inside the spread)",
        [
            "E0 reads its SimulatedLOB (historical snapshot + its own overlay).",
            "FillSimulator: 100 < best ask 101 → NOT marketable.",
            "Order is stored in EngineView0 ONLY — a private diff.",
            "Nobody else sees it; the shared HistoricalLOB is untouched.",
        ],
        {0: f},
    )

    f = sim.submit_limit(1, OrderId(200), "buy", 102.0, 7.0, ts=2)
    emit(
        "engine1", 1, "E1 decision — BUY 7 @ 102 (marketable → fills)",
        [
            "FillSimulator: 102 ≥ best ask 101 → MARKETABLE, walk the book.",
            "Fill 4 @ 101 then 3 @ 102 (fill-at-touch on each level).",
            "Consumed liquidity is recorded in EngineView1 → E1 stops seeing it.",
            "Other engines still see 101/102 in full — shared book unchanged.",
        ],
        {1: f},
    )

    f = sim.submit_limit(2, OrderId(300), "buy", 99.0, 4.0, ts=2)
    emit(
        "engine2", 2, "E2 decision — rest BUY 4 @ 99 (join the queue)",
        [
            "FillSimulator: 99 < best ask 101 → NOT marketable.",
            "Order rests in EngineView2; volume_ahead = 5 (historical 99 queue).",
            "E2's private 99 level becomes 5 (market) + 4 (own) = 9.",
        ],
        {2: f},
    )

    # Phase C — market moves; resting orders are re-evaluated.
    sim.advance(L3Event(3, L3Action.ADD, OrderId(7), "sell", 100.0, 3.0))
    emit(
        "replay", None, "L3 replay — ADD sell 3 @ 100 (market trades down)",
        [
            "New historical SELL at 100 → best ask drops 101 → 100.",
            "The published snapshot is now visible to every engine.",
            "Resting engine orders must be re-checked against the new book.",
        ],
    )

    fills_now: dict[int, list[Fill]] = {}
    for eid in range(N_ENG):
        rf = sim.check_resting_fills(eid)
        if rf:
            fills_now[eid] = rf
    emit(
        "engine0", 0, "E0 resting BUY @ 100 fills at touch",
        [
            "FillSimulator.check_resting_fills() runs for each engine.",
            "E0: best ask 100 ≤ its bid 100 → FILL 3 @ 100.",
            "E1 / E2: their resting prices not yet touched → no fill.",
            "The fill consumes the historical 100 ask in E0's view only.",
        ],
        fills_now,
    )

    sim.advance(L3Event(4, L3Action.ADD, OrderId(8), "sell", 99.0, 5.0))
    emit(
        "replay", None, "L3 replay — ADD sell 5 @ 99 (market reaches 99)",
        [
            "Best ask drops to 99 → E2's resting BUY @ 99 is now touchable.",
            "Snapshot republished; resting orders are re-checked next.",
        ],
    )

    fills_now = {}
    for eid in range(N_ENG):
        rf = sim.check_resting_fills(eid)
        if rf:
            fills_now[eid] = rf
    emit(
        "engine2", 2, "E2 resting BUY @ 99 fills at touch",
        [
            "E2: best ask 99 ≤ its bid 99 → FILL 4 @ 99.",
            "All three engines have now traded from one shared HistoricalLOB,",
            "each via its own private overlay — with no shared mutable state.",
        ],
        fills_now,
    )
    return frames

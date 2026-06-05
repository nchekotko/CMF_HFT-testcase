"""Self-contained matplotlib animation of the multi-engine LOB simulation.

Renders the scenario from :mod:`cmf_mm.simlob.scenario` as a scrubbable HTML
player (play / pause / step / slider — no server, only matplotlib). For a richer,
interactive dashboard use ``scripts/simlob_app.py`` (Streamlit). Each frame shows:

1. **Data-flow diagram** (top) — the active component and the arrows carrying
   data light up: Source -> HistoricalLOB -> BookSnapshot -> EngineView -> Fills.
2. **Order books** (middle) — shared HistoricalLOB + each engine's private book.
   Ghost bars = real market, solid = what the engine sees, hatched = own orders,
   gold star = a fill this step.
3. **Decision log** (bottom) — what happened and where the data went.

Run:  uv run python scripts/simlob_animate.py
Out:  results/simlob/simlob_animation.html
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cmf_mm.simlob.scenario import (
    ENGINE_NAMES,
    N_ENG,
    Frame,
    active_flow,
    build_frames,
)
from cmf_mm.types import Fill, Side

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path("results/simlob/simlob_animation.html")
BID_C, ASK_C, OWN_C, FILL_C = "#2e8b57", "#c0392b", "#1f3a93", "#f1c40f"
HOT, COLD, HOT_TXT = "#f39c12", "#dfe6ee", "#7f8c8d"


# --------------------------------------------------------------------------
# Static layout of the data-flow diagram (axes-fraction coords).
# --------------------------------------------------------------------------
def _flow_boxes() -> dict[str, tuple[float, float, float, float, str]]:
    boxes = {
        "source": (0.02, 0.55, 0.15, 0.30, "Data Source\n(L3 / MBO)"),
        "historical": (0.21, 0.55, 0.16, 0.30, "HistoricalLOB\n(basement)"),
        "snapshot": (0.41, 0.55, 0.16, 0.30, "BookSnapshot\n(published)"),
        "fills": (0.83, 0.55, 0.15, 0.30, "Fills\n(per engine)"),
    }
    for k in range(N_ENG):
        y = 0.06 + k * 0.155
        boxes[f"engine{k}"] = (0.62, y, 0.19, 0.12, f"EngineView{k} -> Sim{k} -> Fill{k}")
    return boxes


_FLOW_ARROWS = [
    ("source", "historical"), ("historical", "snapshot"),
    ("snapshot", "engine0"), ("snapshot", "engine1"), ("snapshot", "engine2"),
    ("engine0", "fills"), ("engine1", "fills"), ("engine2", "fills"),
]


def _draw_flow(ax: Axes, frame: Frame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = _flow_boxes()
    hot_b, hot_a = active_flow(frame)

    def center(name: str) -> tuple[float, float]:
        x, y, w, h, _ = boxes[name]
        return x + w / 2, y + h / 2

    for a, b in _FLOW_ARROWS:
        hot = (a, b) in hot_a
        x0, y0 = center(a)
        x1, y1 = center(b)
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
            color=HOT if hot else "#c4cdd6", lw=2.6 if hot else 1.2,
            shrinkA=22, shrinkB=22, zorder=1,
        ))
    for name, (x, y, w, h, label) in boxes.items():
        hot = name in hot_b
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
            facecolor=HOT if hot else COLD,
            edgecolor="#2c3e50" if hot else "#aab7c4",
            lw=2.2 if hot else 1.0, zorder=2,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold" if hot else "normal",
                color="black" if hot else HOT_TXT, zorder=3)
    ax.set_title("Data flow — highlighted = active this step", fontsize=10)


def _draw_book(
    ax: Axes, bids: list[tuple[float, float]], asks: list[tuple[float, float]],
    *, ghost_b: list[tuple[float, float]] | None = None,
    ghost_a: list[tuple[float, float]] | None = None,
    own: dict[Side, set[float]] | None = None,
    fills: list[Fill] | None = None, title: str = "",
) -> None:
    own = own or {"buy": set(), "sell": set()}
    fill_px = {f.price for f in (fills or [])}
    for levels, color in ((ghost_b, BID_C), (ghost_a, ASK_C)):
        for px, sz in levels or []:
            ax.bar(px, sz, width=0.8, color=color, alpha=0.13, zorder=1)
    sides: list[tuple[list[tuple[float, float]], str, Side]] = [
        (bids, BID_C, "buy"), (asks, ASK_C, "sell")]
    for levels, color, side in sides:
        for px, sz in levels:
            is_own = px in own[side]
            ax.bar(px, sz, width=0.8, color=color, alpha=0.85, zorder=2,
                   edgecolor=OWN_C if is_own else "none",
                   linewidth=2.4 if is_own else 0.0,
                   hatch="///" if is_own else None)
            if px in fill_px:
                ax.scatter(px, sz, marker="*", s=260, color=FILL_C,
                           edgecolor="black", zorder=4)
    allpx = [p for p, _ in bids + asks + (ghost_b or []) + (ghost_a or [])]
    if allpx:
        ax.set_xlim(min(allpx) - 1, max(allpx) + 1)
    ax.set_ylim(0, 11)
    ax.set_title(title, fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(labelsize=7)


def render(frames: list[Frame]) -> tuple[FuncAnimation, Callable[[int], None]]:
    fig = plt.figure(figsize=(13, 9), dpi=96)
    gs = GridSpec(3, N_ENG + 1, figure=fig, height_ratios=[1.15, 1.0, 0.7],
                  hspace=0.45, wspace=0.3)
    ax_flow = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_eng = [fig.add_subplot(gs[1, k + 1]) for k in range(N_ENG)]
    ax_log = fig.add_subplot(gs[2, :])

    def update(i: int) -> None:
        fr = frames[i]
        for a in (ax_flow, ax_hist, ax_log, *ax_eng):
            a.clear()
        _draw_flow(ax_flow, fr)
        hb, ha = list(fr.hist.bids), list(fr.hist.asks)
        _draw_book(ax_hist, hb, ha, title=f"HISTORICAL (shared)\nts={fr.ts}")
        for k in range(N_ENG):
            ef = fr.engines[k]
            _draw_book(ax_eng[k], ef.bids, ef.asks, ghost_b=hb, ghost_a=ha,
                       own=ef.own, fills=ef.fills, title=ENGINE_NAMES[k])
        ax_log.axis("off")
        ax_log.text(0.0, 1.0, f"Step {fr.step + 1}/{len(frames)} — {fr.title}",
                    fontsize=12, fontweight="bold", va="top")
        ax_log.text(0.0, 0.74, "\n".join(f"•  {ln}" for ln in fr.log),
                    fontsize=9.5, va="top", family="monospace")
        fig.suptitle("Multi-engine LOB simulation — watch the data flow & decisions",
                     fontsize=13, y=0.99)

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1400, blit=False  # type: ignore[arg-type]
    )
    return anim, update


def main() -> None:
    frames = build_frames()
    anim, _ = render(frames)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = anim.to_jshtml(default_mode="loop")
    OUT.write_text(html, encoding="utf-8")
    plt.close("all")
    print(f"wrote {OUT}  ({len(frames)} frames)")
    print("open it in a browser: play / pause / step / scrub the slider.")


if __name__ == "__main__":
    main()

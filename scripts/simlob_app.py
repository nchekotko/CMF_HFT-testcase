"""Interactive Streamlit dashboard for the multi-engine LOB simulation (Group 3).

A polished, explorable view of the same scenario as the HTML player: step
through the pipeline, watch where data flows and where decisions are made, hover
the order books for exact values.

Run:  uv run streamlit run scripts/simlob_app.py
"""

from __future__ import annotations

import time

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cmf_mm.simlob.scenario import (
    ENGINE_NAMES,
    FLOW_EDGES,
    Frame,
    active_flow,
    build_frames,
    fmt_px,
)
from cmf_mm.types import Side

# ---- palette (dark) -------------------------------------------------------
BG = "#0e1117"
PANEL = "#161b26"
GRID = "#222a38"
BID = "#26d07c"
ASK = "#ff5d5d"
OWN_LINE = "#5b8dff"
FILL = "#ffd23f"
HOT = "#ffb020"
COLD = "#1b2230"
COLD_LINE = "#2c3545"
MUTED = "#8a93a3"
TEXT = "#e6e9ef"

NODE_LABELS = {
    "source": "Data Source<br><span style='font-size:11px'>L3 / MBO</span>",
    "historical": "HistoricalLOB<br><span style='font-size:11px'>basement · 1 writer</span>",
    "snapshot": "BookSnapshot<br><span style='font-size:11px'>published · immutable</span>",
    "engine0": "EngineView0 → Sim0 → Fill0",
    "engine1": "EngineView1 → Sim1 → Fill1",
    "engine2": "EngineView2 → Sim2 → Fill2",
    "fills": "Fills<br><span style='font-size:11px'>per engine</span>",
}
# (x0, y0, w, h) in a 0..12 × 0..6 canvas. The engine stack sits to the RIGHT of
# the snapshot and the Fills box to the right of the engines, so every arrow runs
# through an empty corridor and never pierces a third box.
NODE_BOX = {
    "source": (0.15, 4.9, 2.2, 0.9),
    "historical": (2.65, 4.9, 2.3, 0.9),
    "snapshot": (5.25, 4.9, 2.3, 0.9),
    "engine0": (7.85, 4.25, 2.45, 0.82),
    "engine1": (7.85, 3.05, 2.45, 0.82),
    "engine2": (7.85, 1.85, 2.45, 0.82),
    "fills": (10.6, 3.05, 1.3, 0.82),
}


@st.cache_data(show_spinner=False)
def get_frames() -> list[Frame]:
    return build_frames()


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, w, h = box
    return x0 + w / 2, y0 + h / 2


def _edge_point(
    box: tuple[float, float, float, float], tx: float, ty: float
) -> tuple[float, float]:
    """Point where the segment from box centre to (tx, ty) crosses the box border."""
    x0, y0, w, h = box
    cx, cy = x0 + w / 2, y0 + h / 2
    dx, dy = tx - cx, ty - cy
    best: float | None = None
    for denom, num in ((dx, x0 - cx), (dx, x0 + w - cx), (dy, y0 - cy), (dy, y0 + h - cy)):
        if denom == 0:
            continue
        t = num / denom
        if t <= 1e-9:
            continue
        px, py = cx + dx * t, cy + dy * t
        if x0 - 1e-6 <= px <= x0 + w + 1e-6 and y0 - 1e-6 <= py <= y0 + h + 1e-6:
            best = t if best is None else min(best, t)
    if best is None:
        return cx, cy
    return cx + dx * best, cy + dy * best


def flow_figure(frame: Frame) -> go.Figure:
    hot_n, hot_e = active_flow(frame)
    fig = go.Figure()

    # Edges first (under the boxes): clean border-to-border, no piercing.
    for a, b in FLOW_EDGES:
        ca, cb = _center(NODE_BOX[a]), _center(NODE_BOX[b])
        ax, ay = _edge_point(NODE_BOX[a], *cb)
        bx, by = _edge_point(NODE_BOX[b], *ca)
        hot = (a, b) in hot_e
        fig.add_annotation(
            x=bx, y=by, ax=ax, ay=ay, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=3.2 if hot else 1.6,
            arrowcolor=HOT if hot else COLD_LINE, opacity=1.0 if hot else 0.55,
        )

    # Boxes.
    for name, box in NODE_BOX.items():
        x0, y0, w, h = box
        hot = name in hot_n
        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x0 + w, y1=y0 + h,
            line=dict(color=HOT if hot else COLD_LINE, width=3 if hot else 1.2),
            fillcolor=HOT if hot else COLD, opacity=0.95 if hot else 0.9,
            layer="below",
        )
        cx, cy = _center(box)
        fig.add_annotation(
            x=cx, y=cy, text=NODE_LABELS[name], showarrow=False,
            font=dict(color="#10131a" if hot else TEXT,
                      size=13 if hot else 12,
                      family="Inter, system-ui, sans-serif"),
            align="center",
        )

    fig.update_xaxes(visible=False, range=[0, 12])
    fig.update_yaxes(visible=False, range=[0, 6])
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False,
    )
    return fig


def _book_traces(
    fig: go.Figure, col: int, bids: list[tuple[float, float]],
    asks: list[tuple[float, float]], ghost_b: list[tuple[float, float]],
    ghost_a: list[tuple[float, float]], own: dict[Side, set[float]],
    fills_px: set[float],
) -> None:
    # Ghost = the real market (faint), drawn behind.
    for levels, color in ((ghost_b, BID), (ghost_a, ASK)):
        if levels:
            xs, ys = zip(*levels, strict=True)
            fig.add_bar(
                x=xs, y=ys, marker_color=color, opacity=0.16, width=0.78,
                showlegend=False, hoverinfo="skip", row=1, col=col,
            )
    # Solid = what this participant sees; own orders get an accent border + hatch.
    for levels, color, side in (
        (bids, BID, "buy"), (asks, ASK, "sell"),
    ):
        if not levels:
            continue
        xs = [p for p, _ in levels]
        ys = [s for _, s in levels]
        own_px = own[side]  # type: ignore[index]
        line_w = [3 if p in own_px else 0 for p in xs]
        patt = ["/" if p in own_px else "" for p in xs]
        fig.add_bar(
            x=xs, y=ys, width=0.78, marker_color=color,
            marker_line_color=OWN_LINE, marker_line_width=line_w,
            marker_pattern_shape=patt, marker_pattern_fgcolor=OWN_LINE,
            marker_pattern_solidity=0.35,
            opacity=0.92, showlegend=False, row=1, col=col,
            customdata=[("own" if p in own_px else "market") for p in xs],
            hovertemplate="px %{x}<br>size %{y}<br>%{customdata}<extra></extra>",
        )
        star_x = [p for p, _ in levels if p in fills_px]
        star_y = [s for p, s in levels if p in fills_px]
        if star_x:
            fig.add_scatter(
                x=star_x, y=star_y, mode="markers",
                marker=dict(symbol="star", size=18, color=FILL,
                            line=dict(color="#1a1d24", width=1)),
                showlegend=False, hoverinfo="skip", row=1, col=col,
            )


def books_figure(frame: Frame, visible: list[int]) -> go.Figure:
    titles = ["HISTORICAL · shared"] + [ENGINE_NAMES[k] for k in visible]
    fig = make_subplots(
        rows=1, cols=len(titles), shared_yaxes=True,
        subplot_titles=titles, horizontal_spacing=0.025,
    )
    hb, ha = list(frame.hist.bids), list(frame.hist.asks)
    # Historical (no ghost/own).
    _book_traces(fig, 1, hb, ha, [], [], {"buy": set(), "sell": set()}, set())
    for i, k in enumerate(visible):
        ef = frame.engines[k]
        fpx = {f.price for f in ef.fills}
        _book_traces(fig, i + 2, ef.bids, ef.asks, hb, ha, ef.own, fpx)

    # Shared x-range across all panels for honest comparison.
    allpx = [p for p, _ in hb + ha]
    for k in visible:
        allpx += [p for p, _ in frame.engines[k].bids + frame.engines[k].asks]
    if allpx:
        lo, hi = min(allpx) - 1, max(allpx) + 1
        fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[0, 11], gridcolor=GRID, zeroline=False)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, dtick=1)
    fig.update_layout(
        barmode="overlay", height=330, bargap=0.12,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=TEXT),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.font.color = TEXT
    return fig


# ---- page -----------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Multi-Engine LOB", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(
        f"""
        <style>
        .stApp {{ background:{BG}; }}
        .block-container {{ padding-top:1.4rem; padding-bottom:1rem; }}
        h1,h2,h3,h4 {{ color:{TEXT}; font-family:Inter,system-ui,sans-serif; }}
        .logcard {{ background:{PANEL}; border:1px solid {GRID};
            border-radius:12px; padding:16px 18px; }}
        .logcard li {{ color:{TEXT}; margin:4px 0; font-size:14px; }}
        .steptag {{ color:{HOT}; font-weight:700; letter-spacing:.04em;
            font-size:13px; text-transform:uppercase; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    frames = get_frames()
    n = len(frames)
    if "step" not in st.session_state:
        st.session_state.step = 0

    # ---- sidebar controls ----
    with st.sidebar:
        st.header("⏯  Controls")
        c1, c2, c3 = st.columns(3)
        if c1.button("◀", use_container_width=True):
            st.session_state.step = max(0, st.session_state.step - 1)
        if c2.button("▶", use_container_width=True):
            st.session_state.step = min(n - 1, st.session_state.step + 1)
        if c3.button("⏮", use_container_width=True):
            st.session_state.step = 0
        st.session_state.step = st.slider(
            "Step", 1, n, st.session_state.step + 1) - 1
        play = st.toggle("Auto-play", value=False)
        speed = st.select_slider(
            "Speed", options=["slow", "normal", "fast"], value="normal")
        st.divider()
        visible = st.multiselect(
            "Engines shown", options=list(range(len(ENGINE_NAMES))),
            default=list(range(len(ENGINE_NAMES))),
            format_func=lambda k: ENGINE_NAMES[k],
        )
        st.divider()
        st.markdown(
            f"<small style='color:{MUTED}'>"
            "★ fill · hatched = engine's own order · faint = real market "
            "(ghost) · solid = what the engine sees</small>",
            unsafe_allow_html=True,
        )

    fr = frames[st.session_state.step]
    if not visible:
        visible = list(range(len(ENGINE_NAMES)))

    # ---- header ----
    st.markdown("## Multi-Engine LOB Simulation")
    st.markdown(
        f"<span class='steptag'>Step {fr.step + 1} / {n}</span>"
        f"&nbsp;&nbsp;<span style='color:{TEXT};font-size:18px;font-weight:600'>"
        f"{fr.title}</span>",
        unsafe_allow_html=True,
    )

    # ---- flow diagram ----
    st.plotly_chart(flow_figure(fr), use_container_width=True,
                    config={"displayModeBar": False})

    # ---- metric cards ----
    cols = st.columns(1 + len(visible))
    h = fr.hist
    cols[0].metric("HISTORICAL  bid / ask",
                   f"{fmt_px(h.best_bid())} / {fmt_px(h.best_ask())}")
    for i, k in enumerate(visible):
        v_bids = fr.engines[k].bids
        v_asks = fr.engines[k].asks
        bb = v_bids[0][0] if v_bids else float("nan")
        ba = v_asks[0][0] if v_asks else float("nan")
        nfills = len(fr.engines[k].fills)
        cols[i + 1].metric(
            ENGINE_NAMES[k], f"{fmt_px(bb)} / {fmt_px(ba)}",
            delta=f"{nfills} fill(s) this step" if nfills else None,
        )

    # ---- order books ----
    st.plotly_chart(books_figure(fr, visible), use_container_width=True,
                    config={"displayModeBar": False})

    # ---- decision log + fills ----
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Decision log")
        items = "".join(f"<li>{ln}</li>" for ln in fr.log)
        st.markdown(f"<div class='logcard'><ul>{items}</ul></div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown("#### Fills this step")
        rows = [
            {"engine": ENGINE_NAMES[k], "side": f.side,
             "size": f.size, "price": f.price, "mid": f.mid_at_fill}
            for k, f in fr.fills_this_step()
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("— no fills on this step —")

    # ---- auto-play ----
    if play and st.session_state.step < n - 1:
        time.sleep({"slow": 1.6, "normal": 0.9, "fast": 0.4}[speed])
        st.session_state.step += 1
        st.rerun()


if __name__ == "__main__":
    main()

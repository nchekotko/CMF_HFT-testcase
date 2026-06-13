"""
Ablation: does trade-volume-imbalance (VI) help, especially at the dead tau=120s?

Builds ONE train matrix with VI features (base 13 + 6 VI), then trains two models per tau:
  - base  = first 13 columns (production features)
  - full  = 13 + VI
picks kept-fraction by time-series CV for each, and compares held-out Feb Score.

VI = signed taker notional over {5,30,120}s (aligned to maker side; raw + volume-normalised),
computed from a causal 1s flow grid (only fully-elapsed seconds before the trade's second).

Usage: python explore_volimb.py [BTC|ETH]
"""
from __future__ import annotations

import sys

import numpy as np

import baseline as B
import ml_baseline as M

NBASE = len(M.FEATURES)


def _train_set(X, y, w, t):
    """CV-pick kept-fraction, then fit on all valid rows. Returns (model, keep, cv_score)."""
    best_q, table = M._cv_pick_fraction(X, y, w, t)
    valid = np.isfinite(y)
    model = M._fit_model(X[valid], y[valid], w[valid])
    return model, best_q, dict(table)[best_q]


def run(sym: str = "BTC", TAUS=M.TAUS_S) -> None:
    print(f"=== VI ablation {sym}  taus={list(TAUS)} ===")
    print("Building TRAIN with VI features (Dec'25-Jan'26, stride 3)...")
    ds = M.build_dataset(B._epoch_us(2025, 12, 1), B._epoch_us(2026, 2, 1),
                         sym, stride_days=3, keep_per_day=M.KEEP_PER_DAY, taus=TAUS, volimb=True)
    X, w, t = ds["X"], ds["w"], ds["t"]
    print(f"train matrix: {X.shape[0]:,} rows, base {NBASE} + VI {X.shape[1]-NBASE} = {X.shape[1]}")

    base, full = {}, {}
    for tau in TAUS:
        y = ds["y"][tau]
        bm, bk, bcv = _train_set(X[:, :NBASE], y, w, t)
        fm, fk, fcv = _train_set(X, y, w, t)
        base[tau] = dict(m=bm, keep=bk, cv=bcv)
        full[tau] = dict(m=fm, keep=fk, cv=fcv)
        print(f"  tau={tau:>3}s  base CV {bcv:+.4f} (keep {bk:.2f}) | "
              f"full CV {fcv:+.4f} (keep {fk:.2f})")

    # --- held-out Feb eval (memory-safe sampled days) ---
    pad_us = max(TAUS) * 1_000_000 + 5_000_000
    lo, hi = B._epoch_us(2026, 2, 1), B._epoch_us(2026, 3, 1)
    acc = {kind: {tau: dict(wp=0.0, w=0.0, kwp=0.0, kw=0.0) for tau in TAUS}
           for kind in ("base", "full")}
    day, n_days = lo, 0
    print("\nEvaluating on Feb (stride 2)...")
    while day < hi:
        res = M._load_eval_day(sym, day, day + M.DAY_US, pad_us, M.EVAL_CAP_ROWS)
        d0 = day
        day += 2 * M.DAY_US
        if res is None:
            continue
        sampled, bbo, liq_bn, liq_by, n_full = res
        if bbo.is_empty():
            continue
        n_days += 1
        cluster = sampled["_cln"].fill_null(1).to_numpy().astype(np.float64)
        grid = M.flow_grid_for_day(sym, d0, d0 + M.DAY_US)
        ctx = M.FeatureContext(sampled, bbo, liq_bn, liq_by,
                               precomputed_cluster=cluster, volimb=True, flow_grid=grid)
        Xf = ctx.build(np.arange(ctx.n))
        wv = ctx.weight_all
        for tau in TAUS:
            pnl = B.markout_pnl_bps(sampled, bbo, tau * 1_000_000)
            valid = np.isfinite(pnl)
            p, ww = pnl[valid], wv[valid]
            for kind, mdl, Xuse in (("base", base, Xf[:, :NBASE]), ("full", full, Xf)):
                pred = mdl[tau]["m"].predict(Xuse)[valid]
                cut = np.quantile(pred, 1.0 - mdl[tau]["keep"])
                ff = (pred < cut).astype(float)
                kw = ww * (1.0 - ff)
                s = acc[kind][tau]
                s["wp"] += float((ww * p).sum()); s["w"] += float(ww.sum())
                s["kwp"] += float((kw * p).sum()); s["kw"] += float(kw.sum())
        print(f"  day {n_days}: {n_full:,} -> {len(sampled):,}")

    print(f"\n=== VI ABLATION ({sym}, val=Feb'26, {n_days} days) ===")
    print(f"  {'tau':>4}  {'base_val':>9}  {'full_val':>9}  {'delta':>8}")
    for tau in TAUS:
        def sc(kind):
            s = acc[kind][tau]
            return s["kwp"] / max(s["kw"], 1e-9) - s["wp"] / max(s["w"], 1e-9)
        b, fv = sc("base"), sc("full")
        print(f"  {tau:>4}  {b:>+9.4f}  {fv:>+9.4f}  {fv - b:>+8.4f}")


if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "BTC"
    taus = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else M.TAUS_S
    run(sym, taus)

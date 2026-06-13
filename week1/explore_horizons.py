"""
Experiment: term structure of the signal across short horizons τ ∈ {1,5,10,30,120,300}s.

Diagnostic only — does NOT touch the submission artifacts (ml_artifacts_<SYM>.joblib).
For each τ it trains a per-τ model (same recipe as ml_baseline), picks the kept-fraction by
forward-chaining time-series CV on the train window, and evaluates Score on held-out Feb.

Expectation: at very short τ, PnL_all ≈ half-spread + rebate (mechanically positive, little
adverse selection), so Score (=PnL_kept−PnL_all) may be small; the edge should grow with τ as
adverse selection appears. This shows WHERE the filterable toxicity lives.

Usage: python explore_horizons.py [BTC|ETH]
"""
from __future__ import annotations

import sys

import numpy as np

import baseline as B
import ml_baseline as M

TAUS_EXP = [1, 5, 10, 30, 120, 300]


def run(sym: str = "BTC") -> None:
    print(f"=== horizon sweep {sym}: tau = {TAUS_EXP}s ===")
    print("Building TRAIN dataset (Dec'25-Jan'26, stride 3 days)...")
    ds = M.build_dataset(B._epoch_us(2025, 12, 1), B._epoch_us(2026, 2, 1),
                         sym, stride_days=3, keep_per_day=M.KEEP_PER_DAY, taus=TAUS_EXP)
    print(f"train matrix: {ds['X'].shape[0]:,} rows x {ds['X'].shape[1]} feats")

    models, keep = {}, {}
    cv_score = {}
    for tau in TAUS_EXP:
        y = ds["y"][tau]
        best_q, table = M._cv_pick_fraction(ds["X"], y, ds["w"], ds["t"])
        cv_score[tau] = dict(table)[best_q]
        valid = np.isfinite(y)
        models[tau] = M._fit_model(ds["X"][valid], y[valid], ds["w"][valid])
        keep[tau] = best_q
        print(f"  tau={tau:>3}s  CV best keep {best_q:.2f}  CV Score {cv_score[tau]:+.4f}")

    # --- held-out eval on Feb (memory-safe sampled days) ---
    pad_us = max(TAUS_EXP) * 1_000_000 + 5_000_000
    lo, hi = B._epoch_us(2026, 2, 1), B._epoch_us(2026, 3, 1)
    acc = {tau: dict(wp=0.0, w=0.0, kwp=0.0, kw=0.0, kw_turn=0.0) for tau in TAUS_EXP}
    day, n_days = lo, 0
    print("\nEvaluating on Feb (stride 2)...")
    while day < hi:
        res = M._load_eval_day(sym, day, day + M.DAY_US, pad_us, M.EVAL_CAP_ROWS)
        day += 2 * M.DAY_US
        if res is None:
            continue
        sampled, bbo, liq_bn, liq_by, n_full = res
        if bbo.is_empty():
            continue
        n_days += 1
        cluster = sampled["_cln"].fill_null(1).to_numpy().astype(np.float64)
        scale = n_full / len(sampled)
        ctx = M.FeatureContext(sampled, bbo, liq_bn, liq_by, precomputed_cluster=cluster)
        w = ctx.weight_all
        preds = ctx.predict_chunked(models)
        for tau in TAUS_EXP:
            pnl = B.markout_pnl_bps(sampled, bbo, tau * 1_000_000)
            cut = np.quantile(preds[tau], 1.0 - keep[tau])
            f = (preds[tau] < cut).astype(float)
            valid = np.isfinite(pnl)
            p, ww, ff = pnl[valid], w[valid], f[valid]
            kw = ww * (1.0 - ff)
            s = acc[tau]
            s["wp"] += float((ww * p).sum());  s["w"] += float(ww.sum())
            s["kwp"] += float((kw * p).sum()); s["kw"] += float(kw.sum())
            s["kw_turn"] += float(kw.sum()) * scale

    print(f"\n=== TERM STRUCTURE ({sym}, val=Feb'26, {n_days} days) ===")
    print(f"  {'tau':>4}  {'keep':>5}  {'CV_Score':>9}  {'val_Score':>9}  "
          f"{'PnL_all':>8}  {'PnL_kept':>9}  {'turn/day(USD)':>15}")
    for tau in TAUS_EXP:
        s = acc[tau]
        pa = s["wp"] / max(s["w"], 1e-9)
        pk = s["kwp"] / max(s["kw"], 1e-9)
        turn = s["kw_turn"] / max(n_days, 1)
        print(f"  {tau:>4}  {keep[tau]:>5.2f}  {cv_score[tau]:>+9.4f}  {pk - pa:>+9.4f}  "
              f"{pa:>+8.3f}  {pk:>+9.3f}  {turn:>15,.0f}")


if __name__ == "__main__":
    run(sys.argv[1].upper() if len(sys.argv) > 1 else "BTC")

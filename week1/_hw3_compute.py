"""Generate the cached numbers the HW3 notebook plots/tabulates:

  hw3_curves.npz   -- event-study reaction curves (same/opp), thresholds {90,95,99}pctl,
                      tau in {30,120,300}, bucket 10s, window 300s, pooled over train days.
  hw3_metrics.json -- Score/PnL/turnover for the candidate filters + Task-2 baseline,
                      on TRAIN and VAL, per tau.

Heavy (loads trades+bbo day-by-day); run in background. The notebook reads the caches.
"""
from __future__ import annotations

import gc
import json

import numpy as np
import polars as pl

import baseline as B
import hw3_liq_filter as H

SYM = "BTC"
TAUS_S = B.TAUS_S
PCTLS = (90, 95, 99)
WINDOWS = (30, 60, 120, 300)
WINDOW_EDA_S = 300
BUCKET_S = 10
N_BUCKETS = WINDOW_EDA_S // BUCKET_S
TRAIN_STRIDE = 3
VAL_STRIDE = 2


def per_symbol_thresholds():
    """USD notional percentile thresholds for each base symbol, from the TRAIN liq table."""
    lo, hi = B._epoch_us(2025, 12, 1), B._epoch_us(2026, 2, 1)  # full train (Dec+Jan)
    out = {}
    for base, fname in (("btcusdt", B.SYM_FILE["BTC"]), ("ethusdt", B.SYM_FILE["ETH"])):
        liq = (pl.scan_parquet(B.DATA / "binance_liquidations" / f"{fname}.parquet")
               .filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))
               .select(["price", "amount"]).collect(engine="streaming"))
        n = (liq["price"] * liq["amount"]).to_numpy()
        out[base] = {q: float(np.percentile(n, q)) for q in PCTLS}
        out[base]["count"] = int(len(n))
        out[base]["max"] = float(n.max())
    return out


def _empty_curves():
    return {q: {t: {"same_sum": np.zeros(N_BUCKETS), "same_cnt": np.zeros(N_BUCKETS),
                    "opp_sum": np.zeros(N_BUCKETS), "opp_cnt": np.zeros(N_BUCKETS)}
                for t in TAUS_S} for q in PCTLS}


def accumulate_curves(curves, trades, bbo, liq_bn, thr_btc, markouts):
    """Event study: per large liq, bucket in-window trades by time-after and direction;
    add their markout (each tau) to same/opp running sums. thr_btc maps pctl->USD."""
    t_tr = trades["timestamp"].to_numpy()
    buy_tr = trades["side"].to_numpy() == "buy"
    bucket_us = BUCKET_S * 1_000_000
    window_us = WINDOW_EDA_S * 1_000_000
    notion = (liq_bn["price"] * liq_bn["amount"]).to_numpy() if not liq_bn.is_empty() else np.array([])
    l_ts_all = liq_bn["timestamp"].to_numpy() if not liq_bn.is_empty() else np.array([])
    l_buy_all = (liq_bn["side"].to_numpy() == "buy") if not liq_bn.is_empty() else np.array([])
    for q in PCTLS:
        keep = notion >= thr_btc[q]
        l_ts, l_buy = l_ts_all[keep], l_buy_all[keep]
        for tliq, lbuy in zip(l_ts, l_buy):
            lo = np.searchsorted(t_tr, tliq, side="left")
            hi = np.searchsorted(t_tr, tliq + window_us, side="right")
            if hi <= lo:
                continue
            b = np.minimum(((t_tr[lo:hi] - tliq) // bucket_us).astype(np.int64), N_BUCKETS - 1)
            same = buy_tr[lo:hi] == lbuy
            for t in TAUS_S:
                pn = markouts[t][lo:hi]
                ok = np.isfinite(pn)
                ms, mo = same & ok, (~same) & ok
                C = curves[q][t]
                if ms.any():
                    np.add.at(C["same_sum"], b[ms], pn[ms]); np.add.at(C["same_cnt"], b[ms], 1.0)
                if mo.any():
                    np.add.at(C["opp_sum"], b[mo], pn[mo]); np.add.at(C["opp_cnt"], b[mo], 1.0)


def candidate_filters(trades, bbo, liq_bn, liq_by, thr_btc, include_bybit_variants=True):
    """{name: 0/1 array} for the grid (Binance-only) + optional Bybit-combined + baseline."""
    out = {}
    for q in PCTLS:
        for w in WINDOWS:
            out[f"same_p{q}_w{w}"] = H.build_filter(
                trades, liq_bn, liq_by, threshold_usd=thr_btc[q],
                window_seconds=w, direction="same", use_bybit=False)
    if include_bybit_variants:  # train-only diagnostic; Bybit was found to hurt -> dropped on val
        for q in (95,):
            for w in (120, 300):
                out[f"same_p{q}_w{w}_bybit"] = H.build_filter(
                    trades, liq_bn, liq_by, threshold_usd=thr_btc[q],
                    window_seconds=w, direction="same", use_bybit=True)
    out["baseline_task2"] = B.classify_trades(trades, bbo, liq_bn, liq_by)[TAUS_S[0]]
    return out


# --- memory-safe sliced loading (for the high-volume Feb days that don't fit whole) ---
LIQ_PAD_US = 300 * 1_000_000  # cover the largest HW3 window so per-slice filters stay exact


def _load_slice(sym, a, b, bbo_pad_us):
    f = B.SYM_FILE[sym]
    trades = B._load_window("binance_trades", f, a, b,
                            ["timestamp", "ticker", "side", "price", "amount"])
    bbo = B._load_window("binance_booktickers", f, a, b + bbo_pad_us,
                         ["timestamp", "bid_price", "ask_price"])
    liq_bn = B._load_window("binance_liquidations", f, a - LIQ_PAD_US, b,
                            ["timestamp", "ticker", "side", "price", "amount"])
    liq_by = B._load_window("bybit_liquidations", B.BYBIT_FILE[sym], a - LIQ_PAD_US, b,
                            ["timestamp", "ticker", "side", "price", "amount"])
    return trades, bbo, liq_bn, liq_by


def metric_pass_chunked(lo, hi, stride, thr_btc, slice_hours=4):
    """Like metric_pass but loads each day in [a,b) time-slices so peak memory stays ~few-M rows.
    Half-open slices tile the day exactly (no double-count) and never split a same-timestamp
    cluster; liqs carry a 300 s look-back pad so window filters are identical to a whole-day load.
    Binance-only grid + baseline (Bybit variants dropped)."""
    bbo_pad_us = max(TAUS_S) * 1_000_000 + 5_000_000
    slice_us = slice_hours * 3600 * 1_000_000
    acc, names, n_days = None, None, 0
    day = lo
    while day < hi:
        d_lo, d_hi = day, day + B.DAY_US
        day += stride * B.DAY_US
        day_has = False
        a = d_lo
        while a < d_hi:
            b = min(a + slice_us, d_hi)
            trades, bbo, liq_bn, liq_by = _load_slice(SYM, a, b, bbo_pad_us)
            a = b
            if trades.is_empty() or bbo.is_empty():
                del trades, bbo, liq_bn, liq_by; gc.collect(); continue
            day_has = True
            weight = np.minimum((trades["price"] * trades["amount"]).to_numpy(), B.CLIP)
            fdict = candidate_filters(trades, bbo, liq_bn, liq_by, thr_btc,
                                      include_bybit_variants=False)
            if names is None:
                names = list(fdict); acc = _zero_metric_acc(names)
            for t in TAUS_S:
                pnl = B.markout_pnl_bps(trades, bbo, t * 1_000_000)
                valid = np.isfinite(pnl)
                p, w = pnl[valid], weight[valid]
                for fn in names:
                    ff = fdict[fn][valid].astype(float)
                    kw, fw = w * (1.0 - ff), w * ff
                    s = acc[fn][t]
                    s["wp"] += float((w*p).sum());  s["w"] += float(w.sum())
                    s["kwp"] += float((kw*p).sum()); s["kw"] += float(kw.sum())
                    s["fwp"] += float((fw*p).sum()); s["fw"] += float(fw.sum())
                    s["keep_n"] += float((1.0-ff).sum()); s["n"] += float(len(p))
                del pnl, valid, p, w
            del trades, bbo, liq_bn, liq_by, weight, fdict; gc.collect()
        if day_has:
            n_days += 1
            print(f"    day {n_days} done (chunked)", flush=True)
    metrics = {}
    for fn in names:
        metrics[fn] = {}
        for t in TAUS_S:
            s = acc[fn][t]
            pa = s["wp"]/max(s["w"],1e-9); pk = s["kwp"]/max(s["kw"],1e-9)
            pf = s["fwp"]/max(s["fw"],1e-9)
            metrics[fn][str(t)] = dict(
                score=pk-pa, pnl_all=pa, pnl_kept=pk, pnl_filtered=pf,
                kept_frac=s["keep_n"]/max(s["n"],1e-9),
                turnover_per_day=s["kw"]/max(n_days,1))
    return metrics, n_days, None


def _zero_metric_acc(names):
    keys = ["wp", "w", "kwp", "kw", "fwp", "fw", "keep_n", "n"]
    return {fn: {t: {k: 0.0 for k in keys} for t in TAUS_S} for fn in names}


def metric_pass(lo, hi, stride, thr_btc, want_curves):
    """Memory-safe day-by-day pass. Markouts are computed ONE tau at a time and freed, so the
    heavy baseline filter (group_by/join on ~20M rows) never coexists with three 150 MB markout
    arrays (that combo OOM'd the Feb days). gc.collect() between days releases polars buffers."""
    pad_us = max(TAUS_S) * 1_000_000 + 5_000_000
    acc, names, n_days = None, None, 0
    curves = _empty_curves() if want_curves else None
    day = lo
    while day < hi:
        trades, bbo, liq_bn, liq_by = B.load_day(SYM, day, day + B.DAY_US, pad_us)
        day += stride * B.DAY_US
        if trades.is_empty() or bbo.is_empty():
            del trades, bbo, liq_bn, liq_by; gc.collect()
            continue
        n_days += 1
        weight = np.minimum((trades["price"] * trades["amount"]).to_numpy(), B.CLIP)

        if want_curves:  # build all-tau markouts briefly, accumulate curves, then free them
            markouts = {t: B.markout_pnl_bps(trades, bbo, t * 1_000_000) for t in TAUS_S}
            liq_bn_sym = H.large_liquidations(liq_bn, "btcusdt", {"btcusdt": 0.0})
            accumulate_curves(curves, trades, bbo, liq_bn_sym, thr_btc, markouts)
            del markouts, liq_bn_sym; gc.collect()

        # filters first (heavy baseline) while no markout arrays are held
        fdict = candidate_filters(trades, bbo, liq_bn, liq_by, thr_btc)
        if names is None:
            names = list(fdict); acc = _zero_metric_acc(names)
        for t in TAUS_S:
            pnl = B.markout_pnl_bps(trades, bbo, t * 1_000_000)
            valid = np.isfinite(pnl)
            p, w = pnl[valid], weight[valid]
            for fn in names:
                ff = fdict[fn][valid].astype(float)
                kw, fw = w * (1.0 - ff), w * ff
                s = acc[fn][t]
                s["wp"] += float((w*p).sum());  s["w"] += float(w.sum())
                s["kwp"] += float((kw*p).sum()); s["kw"] += float(kw.sum())
                s["fwp"] += float((fw*p).sum()); s["fw"] += float(fw.sum())
                s["keep_n"] += float((1.0-ff).sum()); s["n"] += float(len(p))
            del pnl, valid, p, w
        del trades, bbo, liq_bn, liq_by, weight, fdict; gc.collect()
        print(f"    day {n_days} done", flush=True)
    metrics = {}
    for fn in names:
        metrics[fn] = {}
        for t in TAUS_S:
            s = acc[fn][t]
            pa = s["wp"]/max(s["w"],1e-9); pk = s["kwp"]/max(s["kw"],1e-9)
            pf = s["fwp"]/max(s["fw"],1e-9)
            metrics[fn][str(t)] = dict(
                score=pk-pa, pnl_all=pa, pnl_kept=pk, pnl_filtered=pf,
                kept_frac=s["keep_n"]/max(s["n"],1e-9),
                turnover_per_day=s["kw"]/max(n_days,1))
    return metrics, n_days, curves


def save_curves(curves, nd_train, thr_btc):
    save = {"edges": np.arange(N_BUCKETS) * BUCKET_S,
            "pctls": np.array(PCTLS), "taus": np.array(TAUS_S),
            "train_days": np.array([nd_train]),
            "thr_btc": np.array([thr_btc[q] for q in PCTLS])}
    for q in PCTLS:
        for t in TAUS_S:
            C = curves[q][t]
            for k in ("same_sum", "same_cnt", "opp_sum", "opp_cnt"):
                save[f"q{q}_t{t}_{k}"] = C[k]
    np.savez("hw3_curves.npz", **save)


def write_metrics(thr, nd_train, m_train, m_val, nd_val):
    with open("hw3_metrics.json", "w") as fh:
        json.dump({"thresholds": thr, "train_days": nd_train, "val_days": nd_val,
                   "taus": list(TAUS_S), "train": m_train, "val": m_val or {}}, fh, indent=2)


def main():
    import os
    thr = per_symbol_thresholds()
    thr_btc = {q: thr["btcusdt"][q] for q in PCTLS}
    print("per-symbol thresholds:", json.dumps(thr, indent=2))

    # Resume: if a previous run already saved train metrics + curves, skip the (expensive)
    # train pass and go straight to val. This makes iterating on the val pass cheap.
    have_train = (os.path.exists("hw3_metrics.json") and os.path.exists("hw3_curves.npz")
                  and json.load(open("hw3_metrics.json")).get("train"))
    if have_train and os.environ.get("FORCE_TRAIN") != "1":
        cached = json.load(open("hw3_metrics.json"))
        m_train, nd_train = cached["train"], cached["train_days"]
        print(f"[TRAIN] resumed from cache: {nd_train} days, {len(m_train)} filters", flush=True)
    else:
        print(f"\n[TRAIN pass] stride={TRAIN_STRIDE} ...", flush=True)
        m_train, nd_train, curves = metric_pass(
            B._epoch_us(2025, 12, 1), B._epoch_us(2026, 2, 1), TRAIN_STRIDE, thr_btc, want_curves=True)
        print(f"  train days={nd_train}", flush=True)
        # SAVE TRAIN ARTIFACTS NOW (so a val failure can't lose the expensive train pass).
        save_curves(curves, nd_train, thr_btc)
        write_metrics(thr, nd_train, m_train, None, 0)
        del curves; gc.collect()
        print("  saved hw3_curves.npz + hw3_metrics.json (train-only)", flush=True)

    print(f"\n[VAL pass] stride={VAL_STRIDE} (chunked, memory-safe) ...", flush=True)
    m_val, nd_val, _ = metric_pass_chunked(
        B._epoch_us(2026, 2, 1), B._epoch_us(2026, 3, 1), VAL_STRIDE, thr_btc, slice_hours=4)
    print(f"  val days={nd_val}", flush=True)
    write_metrics(thr, nd_train, m_train, m_val, nd_val)
    print("  updated hw3_metrics.json with val", flush=True)

    # quick console summary of the headline filter vs baseline
    print("\n=== Score summary (train | val) ===")
    print(f"  {'filter':>22} | " + " ".join(f"t{t}_tr  t{t}_val" for t in TAUS_S))
    for fn in list(m_train):
        if fn not in m_val:        # Bybit variants are train-only diagnostics
            continue
        row = []
        for t in TAUS_S:
            row.append(f"{m_train[fn][str(t)]['score']:+.3f} {m_val[fn][str(t)]['score']:+.3f}")
        print(f"  {fn:>22} | " + "  ".join(row))


if __name__ == "__main__":
    main()

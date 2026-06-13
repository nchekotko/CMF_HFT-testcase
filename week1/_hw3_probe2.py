"""Probe 2 — decisive numbers:
(1) mid-price path after large buy vs sell liquidations (sanity on continuation/reversal);
(2) Score(tau) for the PRESCRIBED filter (filter trades SAME-direction as liq within window)
    and its INVERSE (filter OPPOSITE-direction), over a few sampled days, threshold = 95th pctl.
"""
from __future__ import annotations

import numpy as np
import polars as pl

import baseline as B

SYM = "BTC"
TAUS_S = B.TAUS_S


def large_liq(liq: pl.DataFrame, thr_usd: float) -> pl.DataFrame:
    return liq.with_columns((pl.col("price") * pl.col("amount")).alias("_n")).filter(
        pl.col("_n") >= thr_usd
    )


def same_dir_filter(trades, liq_large, window_us, same=True):
    """f_i=1 if a large liq with side matching (same) / opposing (not same) the trade's
    taker side occurred in [t_i - window_us, t_i]."""
    n = len(trades)
    f = np.zeros(n, dtype=np.int8)
    if liq_large.is_empty():
        return f
    t_tr = trades["timestamp"].to_numpy()
    buy_tr = trades["side"].to_numpy() == "buy"
    for liq_side_buy in (True, False):
        sub = liq_large.filter(pl.col("side") == ("buy" if liq_side_buy else "sell"))
        if sub.is_empty():
            continue
        lt = np.sort(sub["timestamp"].to_numpy())
        # trade matched if any liq in [t_i - window, t_i]  <=>  count in (t_i-window, t_i] > 0
        hi = np.searchsorted(lt, t_tr, side="right")
        lo = np.searchsorted(lt, t_tr - window_us, side="right")
        has = (hi - lo) > 0
        # which trades' taker side we mark depends on same/opposite
        want_buy = liq_side_buy if same else (not liq_side_buy)
        sel = has & (buy_tr == want_buy)
        f[sel] = 1
    return f


def price_path(trades, bbo, liq_large):
    """Avg mid return (bps) at +30/120/300s after each large liq, signed by liq side
    (+ = move in liq direction). Positive => continuation; negative => reversal."""
    if liq_large.is_empty():
        return {}
    b_ts = bbo["timestamp"].to_numpy()
    b_mid = (bbo["bid_price"].to_numpy() + bbo["ask_price"].to_numpy()) * 0.5
    lt = liq_large["timestamp"].to_numpy()
    lsign = np.where(liq_large["side"].to_numpy() == "buy", 1.0, -1.0)

    def mid_at(ts):
        idx = np.searchsorted(b_ts, ts, side="right") - 1
        valid = (idx >= 0) & (ts >= b_ts[0]) & (ts <= b_ts[-1])
        return np.where(valid, b_mid[np.clip(idx, 0, len(b_mid) - 1)], np.nan)

    m0 = mid_at(lt)
    out = {}
    for h in (30, 120, 300):
        mh = mid_at(lt + h * 1_000_000)
        r = lsign * (mh - m0) / m0 * 1e4
        out[h] = r[np.isfinite(r)]
    return out


def main():
    lo, hi = B._epoch_us(2025, 12, 1), B._epoch_us(2026, 1, 1)
    pad_us = 300 * 1_000_000 + 30 * 1_000_000 + 5_000_000

    liq_all = (
        pl.scan_parquet(B.DATA / "binance_liquidations" / f"{B.SYM_FILE[SYM]}.parquet")
        .filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))
        .select(["timestamp", "side", "price", "amount"]).collect(engine="streaming")
    )
    nloc = (liq_all["price"] * liq_all["amount"]).to_numpy()
    THR = float(np.percentile(nloc, 95))
    print(f"threshold 95th pctl = ${THR:,.0f}")

    windows = [30, 60, 120, 300]
    # accumulators for Score for SAME and OPP filter, per window, per tau
    acc = {mode: {w: {t: dict(wp=0., w=0., kwp=0., kw=0., fwp=0., fw=0., keep_n=0., n=0.)
                      for t in TAUS_S} for w in windows}
           for mode in ("same", "opp")}
    path_acc = {h: [] for h in (30, 120, 300)}

    day, n_days = lo, 0
    while day < hi and n_days < 8:
        trades, bbo, liq_bn, _ = B.load_day(SYM, day, day + B.DAY_US, pad_us)
        day += 4 * B.DAY_US
        if trades.is_empty() or bbo.is_empty():
            continue
        n_days += 1
        liq_L = large_liq(liq_bn, THR)
        for h, arr in price_path(trades, bbo, liq_L).items():
            path_acc[h].append(arr)

        weight = np.minimum((trades["price"] * trades["amount"]).to_numpy(), B.CLIP)
        pnls = {t: B.markout_pnl_bps(trades, bbo, t * 1_000_000) for t in TAUS_S}
        for mode in ("same", "opp"):
            for w in windows:
                f = same_dir_filter(trades, liq_L, w * 1_000_000, same=(mode == "same"))
                for t in TAUS_S:
                    pnl = pnls[t]
                    valid = np.isfinite(pnl)
                    p, ww, ff = pnl[valid], weight[valid], f[valid].astype(float)
                    kw, fw = ww * (1 - ff), ww * ff
                    s = acc[mode][w][t]
                    s["wp"] += (ww * p).sum(); s["w"] += ww.sum()
                    s["kwp"] += (kw * p).sum(); s["kw"] += kw.sum()
                    s["fwp"] += (fw * p).sum(); s["fw"] += fw.sum()
                    s["keep_n"] += (1 - ff).sum(); s["n"] += len(p)

    print(f"\n=== mid-price path after large liqs (signed by liq side; + = continuation) "
          f"({n_days} days) ===")
    for h in (30, 120, 300):
        a = np.concatenate(path_acc[h]) if path_acc[h] else np.array([])
        print(f"  +{h:>3}s : mean {a.mean():+.3f} bps   median {np.median(a):+.3f}   n={len(a)}")

    print(f"\n=== Score(tau) by filter mode & window (threshold 95th, {n_days} days) ===")
    for mode in ("same", "opp"):
        print(f"\n  [{mode}-direction filter]  (prescribed = same)")
        print(f"    {'win(s)':>6} {'tau':>4} {'Score':>8} {'PnL_all':>8} {'PnL_kept':>9} "
              f"{'PnL_filt':>9} {'kept%':>6} {'turn/day':>13}")
        for w in windows:
            for t in TAUS_S:
                s = acc[mode][w][t]
                pa = s["wp"]/max(s["w"],1e-9); pk = s["kwp"]/max(s["kw"],1e-9)
                pf = s["fwp"]/max(s["fw"],1e-9)
                print(f"    {w:>6} {t:>4} {pk-pa:>+8.3f} {pa:>+8.3f} {pk:>+9.3f} {pf:>+9.3f} "
                      f"{s['keep_n']/max(s['n'],1e-9)*100:>5.1f}% {s['kw']/n_days:>13,.0f}")


if __name__ == "__main__":
    main()

"""Quick probe: liquidation notional scale + same/opposite reaction curves on a few days.

Confirms the core HW3 premise (continuation vs reversal) before building the notebook.
Reuses baseline.py loaders/markout. BTC only, a few sampled days.
"""
from __future__ import annotations

import numpy as np
import polars as pl

import baseline as B

SYM = "BTC"
WINDOW_S = 300
TAU_EDA_US = 30 * 1_000_000
BUCKET_S = 10
N_BUCKETS = WINDOW_S // BUCKET_S
WINDOW_US = WINDOW_S * 1_000_000
BUCKET_US = BUCKET_S * 1_000_000


def liq_notional(liq: pl.DataFrame) -> np.ndarray:
    return (liq["price"] * liq["amount"]).to_numpy()


def reaction_accumulate(trades, bbo, liq, thr_usd, acc):
    """Event-study accumulation: for each large liq, add 30s markout of in-window trades
    to per-(direction, time-bucket) running sums. acc = dict with sum/cnt arrays."""
    if liq.is_empty():
        return
    liq = liq.with_columns((pl.col("price") * pl.col("amount")).alias("_n")).filter(
        pl.col("_n") >= thr_usd
    )
    if liq.is_empty():
        return
    t_tr = trades["timestamp"].to_numpy()
    side_tr = (trades["side"].to_numpy() == "buy")  # True = taker buy = +1
    # 30s maker markout (with rebate) for every trade, once.
    pnl = B.markout_pnl_bps(trades, bbo, TAU_EDA_US)

    l_ts = liq["timestamp"].to_numpy()
    l_buy = (liq["side"].to_numpy() == "buy")
    for tliq, lbuy in zip(l_ts, l_buy):
        lo = np.searchsorted(t_tr, tliq, side="left")
        hi = np.searchsorted(t_tr, tliq + WINDOW_US, side="right")
        if hi <= lo:
            continue
        off = t_tr[lo:hi] - tliq
        b = np.minimum((off // BUCKET_US).astype(int), N_BUCKETS - 1)
        same = side_tr[lo:hi] == lbuy          # trade taker side matches liq side
        pn = pnl[lo:hi]
        ok = np.isfinite(pn)
        for mask, key in ((same, "same"), (~same, "opp")):
            m = mask & ok
            if not m.any():
                continue
            np.add.at(acc[key]["sum"], b[m], pn[m])
            np.add.at(acc[key]["cnt"], b[m], 1.0)


def main():
    lo, hi = B._epoch_us(2025, 12, 1), B._epoch_us(2026, 1, 1)
    pad_us = WINDOW_US + TAU_EDA_US + 5_000_000

    # --- global notional percentiles over the whole train liq table (small) ---
    liq_all = (
        pl.scan_parquet(B.DATA / "binance_liquidations" / f"{B.SYM_FILE[SYM]}.parquet")
        .filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))
        .select(["timestamp", "side", "price", "amount"])
        .collect(engine="streaming")
    )
    n = liq_notional(liq_all)
    print(f"train binance liq count={len(n):,}  side buy%={100*(liq_all['side']=='buy').mean():.1f}")
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f"  pctl {q:>5}: ${np.percentile(n, q):>14,.0f}")
    print(f"  max: ${n.max():,.0f}   mean: ${n.mean():,.0f}")

    thresholds = {q: float(np.percentile(n, q)) for q in (90, 95, 99)}

    accs = {q: {"same": {"sum": np.zeros(N_BUCKETS), "cnt": np.zeros(N_BUCKETS)},
                "opp": {"sum": np.zeros(N_BUCKETS), "cnt": np.zeros(N_BUCKETS)}}
            for q in thresholds}

    day, n_days = lo, 0
    while day < hi and n_days < 6:
        trades, bbo, liq_bn, _ = B.load_day(SYM, day, day + B.DAY_US, pad_us)
        day += 5 * B.DAY_US
        if trades.is_empty() or bbo.is_empty():
            continue
        n_days += 1
        for q, thr in thresholds.items():
            reaction_accumulate(trades, bbo, liq_bn, thr, accs[q])

    print(f"\n=== reaction curves ({SYM}, {n_days} sampled days), maker PnL bps (incl rebate), tau=30s ===")
    for q, thr in thresholds.items():
        print(f"\n  threshold = {q}th pctl  (${thr:,.0f})")
        print(f"    {'t_after(s)':>10} {'same_mean':>10} {'same_n':>8} {'opp_mean':>10} {'opp_n':>8}")
        for b in range(N_BUCKETS):
            s, o = accs[q]["same"], accs[q]["opp"]
            sm = s["sum"][b] / max(s["cnt"][b], 1)
            om = o["sum"][b] / max(o["cnt"][b], 1)
            print(f"    {b*BUCKET_S:>10} {sm:>+10.3f} {int(s['cnt'][b]):>8} {om:>+10.3f} {int(o['cnt'][b]):>8}")


if __name__ == "__main__":
    main()

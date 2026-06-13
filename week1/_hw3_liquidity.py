"""Liquidity-recovery event study: top-of-book depth & spread as a function of time after a
large Binance liquidation, pooled over train days. Saves hw3_liquidity.npz + a PNG.

Liquidity proxies (only BBO is available, no full book):
  * top-of-book depth in USD = bid_amt*bid_px + ask_amt*ask_px   (and split hit-/opposite-side)
  * spread in bps = (ask_px - bid_px)/mid * 1e4                  (inverse liquidity)

A buy-liquidation consumes ASK liquidity (forced buying lifts offers); a sell-liquidation
consumes BID. "hit side" = the consumed side, which is what depletes and then refills.
"""
from __future__ import annotations

import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import baseline as B

SYM = "BTC"
PCTLS = (95, 99)
TRAIN_LO, TRAIN_HI = B._epoch_us(2025, 12, 1), B._epoch_us(2026, 2, 1)
STRIDE = 3
# offsets after the liquidation (seconds); negatives = pre-event baseline, dense near 0
OFFS_S = np.array([-30, -10, -5, -2, -1, 0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240, 300], float)
OFF_US = (OFFS_S * 1_000_000).astype(np.int64)
PRE_PAD_US = 35 * 1_000_000
POST_PAD_US = 305 * 1_000_000


def thresholds():
    liq = (pl.scan_parquet(B.DATA / "binance_liquidations" / f"{B.SYM_FILE[SYM]}.parquet")
           .filter((pl.col("timestamp") >= TRAIN_LO) & (pl.col("timestamp") < TRAIN_HI))
           .select(["price", "amount"]).collect(engine="streaming"))
    n = (liq["price"] * liq["amount"]).to_numpy()
    return {q: float(np.percentile(n, q)) for q in PCTLS}


def _empty():
    z = lambda: np.zeros(len(OFFS_S))
    return {q: {k: z() for k in ("total", "hit", "opp", "spread", "cnt")} for q in PCTLS}


def accumulate(acc, bbo, liq, thr):
    if bbo.is_empty() or liq.is_empty():
        return
    b_ts = bbo["timestamp"].to_numpy()
    b_bidp = bbo["bid_price"].to_numpy(); b_bida = bbo["bid_amount"].to_numpy()
    b_askp = bbo["ask_price"].to_numpy(); b_aska = bbo["ask_amount"].to_numpy()
    notion = (liq["price"] * liq["amount"]).to_numpy()
    l_ts = liq["timestamp"].to_numpy()
    l_buy = liq["side"].to_numpy() == "buy"
    for q in PCTLS:
        keep = notion >= thr[q]
        lt, lb = l_ts[keep], l_buy[keep]
        if len(lt) == 0:
            continue
        for k, off in enumerate(OFF_US):
            target = lt + off
            idx = np.searchsorted(b_ts, target, side="right") - 1
            valid = (idx >= 0) & (target >= b_ts[0]) & (target <= b_ts[-1])
            if not valid.any():
                continue
            i = np.clip(idx, 0, len(b_ts) - 1)
            bp, ba, ap, aa = b_bidp[i], b_bida[i], b_askp[i], b_aska[i]
            mid = 0.5 * (bp + ap)
            bid_usd = ba * bp
            ask_usd = aa * ap
            total = bid_usd + ask_usd
            hit = np.where(lb, ask_usd, bid_usd)      # buy-liq hits ask; sell-liq hits bid
            opp = np.where(lb, bid_usd, ask_usd)
            spread = (ap - bp) / mid * 1e4
            v = valid
            acc[q]["total"][k] += total[v].sum()
            acc[q]["hit"][k] += hit[v].sum()
            acc[q]["opp"][k] += opp[v].sum()
            acc[q]["spread"][k] += spread[v].sum()
            acc[q]["cnt"][k] += v.sum()


def main():
    thr = thresholds()
    print("thresholds USD:", {q: round(v) for q, v in thr.items()}, flush=True)
    acc = _empty()
    day, n_days = TRAIN_LO, 0
    while day < TRAIN_HI:
        d_lo, d_hi = day, day + B.DAY_US
        day += STRIDE * B.DAY_US
        bbo = B._load_window("binance_booktickers", B.SYM_FILE[SYM],
                             d_lo - PRE_PAD_US, d_hi + POST_PAD_US,
                             ["timestamp", "bid_price", "bid_amount", "ask_price", "ask_amount"])
        liq = B._load_window("binance_liquidations", B.SYM_FILE[SYM], d_lo, d_hi,
                             ["timestamp", "side", "price", "amount"])
        if bbo.is_empty():
            del bbo, liq; gc.collect(); continue
        n_days += 1
        accumulate(acc, bbo, liq, thr)
        del bbo, liq; gc.collect()
        print(f"  day {n_days} done", flush=True)

    # save curves
    save = {"offs_s": OFFS_S, "n_days": np.array([n_days]),
            "thr": np.array([thr[q] for q in PCTLS]), "pctls": np.array(PCTLS)}
    means = {}
    for q in PCTLS:
        c = np.maximum(acc[q]["cnt"], 1.0)
        means[q] = {k: acc[q][k] / c for k in ("total", "hit", "opp", "spread")}
        means[q]["cnt"] = acc[q]["cnt"]
        for k in means[q]:
            save[f"q{q}_{k}"] = means[q][k]
    np.savez("hw3_liquidity.npz", **save)
    print(f"saved hw3_liquidity.npz ({n_days} days)", flush=True)

    # ---- plot ----
    pre = OFFS_S < 0
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    q = 99
    base_total = means[q]["total"][pre].mean()
    ax[0].axhline(base_total / 1e3, color="grey", ls=":", lw=1, label="pre-event level")
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].plot(OFFS_S, means[q]["total"] / 1e3, "-o", ms=3, color="#4C72B0", label="total BBO depth")
    ax[0].plot(OFFS_S, means[q]["hit"] / 1e3, "-o", ms=3, color="#C44E52", label="hit side (consumed)")
    ax[0].plot(OFFS_S, means[q]["opp"] / 1e3, "-o", ms=3, color="#55A868", label="opposite side")
    ax[0].set_xlabel("seconds after liquidation"); ax[0].set_ylabel("top-of-book depth (USD thousands)")
    ax[0].set_title(f"Liquidity (depth) recovery — {q}th-pctl liqs (>= ${thr[q]:,.0f})")
    ax[0].legend(fontsize=8)

    for q, col in ((95, "#DD8452"), (99, "#8172B3")):
        base_sp = means[q]["spread"][pre].mean()
        ax[1].plot(OFFS_S, means[q]["spread"], "-o", ms=3, color=col,
                   label=f"{q}th pctl (pre={base_sp:.2f} bps)")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("seconds after liquidation"); ax[1].set_ylabel("spread (bps)")
    ax[1].set_title("Spread (inverse liquidity) after liquidation")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"BTC liquidity reaction to large liquidations (train, {n_days} days)", y=1.02)
    plt.tight_layout()
    fig.savefig("hw3_liquidity.png", dpi=120, bbox_inches="tight")
    print("saved hw3_liquidity.png", flush=True)

    # console summary
    print("\noffset_s |  depth99(k$)  hit99(k$)  opp99(k$)  spread95  spread99  n99")
    for k, o in enumerate(OFFS_S):
        m99 = means[99]; m95 = means[95]
        print(f"  {o:>6.0f} | {m99['total'][k]/1e3:>10.1f} {m99['hit'][k]/1e3:>9.1f} "
              f"{m99['opp'][k]/1e3:>9.1f} {m95['spread'][k]:>8.3f} {m99['spread'][k]:>8.3f} "
              f"{int(m99['cnt'][k]):>6}")


if __name__ == "__main__":
    main()

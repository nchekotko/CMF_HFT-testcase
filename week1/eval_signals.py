"""Compare candidate filter signals in one day-by-day pass (shares the expensive load).

Computes features once per day, derives several named filters, accumulates exact weighted
sums, and prints a Score/turnover table per filter for TRAIN and VAL. Use this to pick the
signal that actually generalises before baking it into classify_trades.
"""
from __future__ import annotations

import numpy as np

import baseline as B

DAY_US = B.DAY_US
TAUS_S = B.TAUS_S


def candidate_filters(trades, liq_bn, liq_by) -> dict[str, np.ndarray]:
    """Return {name: 0/1 filter array} for every candidate, computed from the same features."""
    ts = trades["timestamp"].to_numpy()
    sgn = np.where(trades["side"].to_numpy() == "buy", 1.0, -1.0)

    p_by = B._signed_liq_pressure(ts, liq_by, B.BYBIT_LOOKBACK_US,
                                  gate_us=B.BYBIT_VISIBILITY_US,
                                  visible_shift_us=B.BYBIT_VISIBILITY_US)
    p_bn = B._signed_liq_pressure(ts, liq_bn, B.BINANCE_LOOKBACK_US, gate_us=0)
    cluster = B._cluster_size(trades)

    rev_by = (np.sign(p_by) == -sgn) & (p_by != 0)
    rev_bn = (np.sign(p_bn) == -sgn) & (p_bn != 0)
    # momentum = opposite sign convention, as a sanity check on the reversal direction
    mom_bn = (np.sign(p_bn) == sgn) & (p_bn != 0)

    return {
        "cluster>=25": cluster >= 25,
        "cluster>=50": cluster >= 50,
        "cluster>=100": cluster >= 100,
        "bybit_rev": rev_by,
        "binance_rev": rev_bn,
        "binance_mom": mom_bn,
        "cluster50|binance_rev": (cluster >= 50) | rev_bn,
        "ensemble(all)": (cluster >= 50) | rev_by | rev_bn,
    }


def run(name: str, lo: int, hi: int, sym: str = "BTC", stride_days: int = 2) -> None:
    pad_us = max(TAUS_S) * 1_000_000 + 5_000_000
    names = None
    acc: dict = {}          # acc[fname][tau] -> sums
    n_days = 0
    day = lo
    while day < hi:
        trades, bbo, liq_bn, liq_by = B.load_day(sym, day, day + DAY_US, pad_us)
        day += stride_days * DAY_US
        if trades.is_empty() or bbo.is_empty():
            continue
        n_days += 1
        weight = np.minimum((trades["price"] * trades["amount"]).to_numpy(), B.CLIP)
        filters = candidate_filters(trades, liq_bn, liq_by)
        if names is None:
            names = list(filters)
            acc = {fn: {t: dict(wp=0.0, w=0.0, kwp=0.0, kw=0.0, keep_n=0.0, n=0.0)
                        for t in TAUS_S} for fn in names}
        for tau in TAUS_S:
            pnl = B.markout_pnl_bps(trades, bbo, tau * 1_000_000)
            valid = np.isfinite(pnl)
            p, w = pnl[valid], weight[valid]
            for fn in names:
                f = filters[fn][valid].astype(float)
                kw = w * (1.0 - f)
                s = acc[fn][tau]
                s["wp"] += float((w * p).sum());   s["w"] += float(w.sum())
                s["kwp"] += float((kw * p).sum()); s["kw"] += float(kw.sum())
                s["keep_n"] += float((1.0 - f).sum()); s["n"] += float(len(p))

    print(f"\n========== {name}  ({sym}, {n_days} days) ==========")
    for fn in names:
        print(f"\n  [{fn}]")
        print(f"    {'tau':>4}  {'Score':>8}  {'PnL_all':>8}  {'PnL_kept':>9}  "
              f"{'kept%':>6}  {'turn/day':>14}")
        for tau in TAUS_S:
            s = acc[fn][tau]
            pa = s["wp"] / max(s["w"], 1e-9)
            pk = s["kwp"] / max(s["kw"], 1e-9)
            print(f"    {tau:>4}  {pk - pa:>+8.3f}  {pa:>+8.3f}  {pk:>+9.3f}  "
                  f"{s['keep_n']/max(s['n'],1e-9)*100:>5.1f}%  {s['kw']/n_days:>14,.0f}")


if __name__ == "__main__":
    run("TRAIN", B._epoch_us(2025, 12, 1), B._epoch_us(2026, 1, 1), stride_days=2)
    run("VAL", B._epoch_us(2026, 2, 1), B._epoch_us(2026, 3, 1), stride_days=2)

"""
Week-1 baseline — liquidation signal that filters Binance maker trades.

Task ("Сигнал с ликвидаций"): we assume we passively collect Binance taker flow as a
maker. For each trade we decide keep (0) or filter (1) so that the maker PnL on the
*kept* trades beats the PnL on *all* trades, while keeping >= $500k/day of clipped
turnover.

Submission entry point
----------------------
    classify_trades(trades, bbo, liq_binance, liq_bybit) -> dict[int, np.ndarray]

Returns, for each horizon tau in (30, 120, 300) seconds, a 0/1 array of length
len(trades): 1 = filter the trade out, 0 = keep it. The function is a pure function of
the four input frames (no look-ahead, no labels), so it runs unchanged on the hidden
test set.

The baseline is a simple ensemble of three documented signals, each established
out-of-sample during exploration (see findings in week1_exploration.ipynb):

  A. Bybit-liquidation reversal (5 s lookback, +200 ms cross-exchange visibility gate).
  B. Binance-liquidation reversal (60 s lookback).
     Liquidations are a *reversal* signal here: after a burst of buy-liquidations the
     mid mean-reverts DOWN over the next minute, so a taker-sell (maker buys) into that
     burst is adversely selected -> filter it.
  C. Sweeper clusters: when many same-side maker fills share one microsecond, one urgent
     aggressor swept the book -> the maker on the other side is adversely selected.

Run as a script to evaluate on the BTC train/val splits and print the metric table
(Score, PnL_kept, PnL_filtered, kept turnover/day) and the turnover-constraint check.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

# ----------------------------------------------------------------------------------
# Task constants
# ----------------------------------------------------------------------------------
TAUS_S = (30, 120, 300)                 # markout horizons (seconds)
CLIP = 100_000.0                        # weight clip w_i = min(notional_i, 100k) USD
REBATE_BPS = 0.5                        # maker rebate added to every fill
TURNOVER_MIN = 500_000.0                # constraint: kept clipped turnover >= 500k USD/day
BYBIT_VISIBILITY_US = 200_000           # +200 ms before a Bybit event is usable to us

# Baseline signal hyper-parameters (fixed; chosen on train during exploration).
BYBIT_LOOKBACK_US = 5_000_000           # 5 s
BINANCE_LOOKBACK_US = 60_000_000        # 60 s
PRESSURE_MIN_USD = 0.0                  # any liq in the window counts
CLUSTER_MIN = 50                        # same-us same-side cluster size that flags a sweeper


# ----------------------------------------------------------------------------------
# Signal building blocks (pure functions of the input frames)
# ----------------------------------------------------------------------------------
def _signed_liq_pressure(
    trade_ts: np.ndarray,
    liq: pl.DataFrame,
    lookback_us: int,
    gate_us: int,
    visible_shift_us: int = 0,
) -> np.ndarray:
    """Signed notional of liquidations visible in (t - lookback, t - gate] for each trade.

    side == "buy" liquidation -> forced buying -> +pressure (upward). The optional
    visible_shift_us models cross-exchange latency (Bybit events arrive +200 ms late).
    Returns zeros if the liquidation frame is empty.
    """
    out = np.zeros(len(trade_ts), dtype=np.float64)
    if liq.is_empty():
        return out
    liq = liq.with_columns(
        pl.when(pl.col("side") == "buy")
        .then(pl.col("price") * pl.col("amount"))
        .otherwise(-pl.col("price") * pl.col("amount"))
        .alias("_sn"),
        (pl.col("timestamp") + visible_shift_us).alias("_ts_vis"),
    ).sort("_ts_vis")

    ts_v = liq["_ts_vis"].to_numpy()
    sn = liq["_sn"].to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(sn)])
    idx_hi = np.searchsorted(ts_v, trade_ts - gate_us, side="right")
    idx_lo = np.searchsorted(ts_v, trade_ts - lookback_us, side="right")
    return csum[idx_hi] - csum[idx_lo]


def _cluster_size(trades: pl.DataFrame) -> np.ndarray:
    """Number of trades sharing each trade's (timestamp, side) — the sweeper-cluster size.

    A pure function of the trades frame; preserves the original row order.
    """
    indexed = trades.select(["timestamp", "side"]).with_row_index("_i")
    sizes = indexed.group_by(["timestamp", "side"]).agg(pl.len().alias("_n"))
    joined = indexed.join(sizes, on=["timestamp", "side"], how="left").sort("_i")
    return joined["_n"].to_numpy()


def _restrict_to_trade_symbols(
    trades: pl.DataFrame, liq: pl.DataFrame, strip_perp: bool
) -> pl.DataFrame:
    """Keep only liquidation rows whose ticker matches the symbol(s) traded.

    Binance tickers look like 'perp:btcusdt'; Bybit liq tickers like 'btcusdt'.
    No-op when the frame is already single-symbol or has no 'ticker' column.
    """
    if liq.is_empty() or "ticker" not in liq.columns or "ticker" not in trades.columns:
        return liq
    bases = {t.split(":")[-1] for t in trades["ticker"].unique().to_list()}
    if strip_perp:
        return liq.filter(
            pl.col("ticker").str.split(":").list.last().is_in(list(bases))
        )
    return liq.filter(pl.col("ticker").is_in(list(bases)))


# ----------------------------------------------------------------------------------
# Submission entry point
# ----------------------------------------------------------------------------------
def classify_trades(
    trades: pl.DataFrame,
    bbo: pl.DataFrame,            # accepted for signature compatibility; baseline doesn't use it
    liq_binance: pl.DataFrame,
    liq_bybit: pl.DataFrame,
) -> dict[int, np.ndarray]:
    """Return {tau_s: 0/1 filter array} for tau_s in (30, 120, 300).

    1 = filter the trade out, 0 = keep it. The baseline signal is horizon-independent,
    so the three arrays are identical; the per-tau dict matches the required format.
    """
    n = len(trades)
    if n == 0:
        empty = np.zeros(0, dtype=np.int8)
        return {tau: empty.copy() for tau in TAUS_S}

    liq_bn = _restrict_to_trade_symbols(trades, liq_binance, strip_perp=True)
    liq_by = _restrict_to_trade_symbols(trades, liq_bybit, strip_perp=True)

    ts = trades["timestamp"].to_numpy()
    sgn = np.where(trades["side"].to_numpy() == "buy", 1.0, -1.0)  # +1 taker buy / maker sell

    # A. Bybit reversal (short lookback, respects the +200 ms visibility gate).
    p_by = _signed_liq_pressure(
        ts, liq_by, BYBIT_LOOKBACK_US, gate_us=BYBIT_VISIBILITY_US,
        visible_shift_us=BYBIT_VISIBILITY_US,
    )
    # B. Binance reversal (longer lookback, no cross-exchange gate).
    p_bn = _signed_liq_pressure(ts, liq_bn, BINANCE_LOOKBACK_US, gate_us=0)

    # Reversal: filter when the taker side runs opposite to recent liq pressure, i.e. the
    # taker is trading in the direction the mid is about to mean-revert -> maker loses.
    rev_by = (np.sign(p_by) == -sgn) & (np.abs(p_by) >= PRESSURE_MIN_USD) & (p_by != 0)
    rev_bn = (np.sign(p_bn) == -sgn) & (np.abs(p_bn) >= PRESSURE_MIN_USD) & (p_bn != 0)

    # C. Sweeper clusters.
    cluster = _cluster_size(trades) >= CLUSTER_MIN

    filt = (rev_by | rev_bn | cluster).astype(np.int8)
    return {tau: filt.copy() for tau in TAUS_S}


# ----------------------------------------------------------------------------------
# Evaluation: markout PnL + metrics (used offline, NOT part of the submission)
# ----------------------------------------------------------------------------------
def markout_pnl_bps(trades: pl.DataFrame, bbo: pl.DataFrame, tau_us: int) -> np.ndarray:
    """Per-trade maker PnL in bps at horizon tau (incl. +0.5 rebate); NaN if t+tau is
    beyond the available BBO. Uses forward-filled (last-observed) Binance mid at t+tau."""
    ts = trades["timestamp"].to_numpy()
    price = trades["price"].to_numpy()
    sgn = np.where(trades["side"].to_numpy() == "buy", 1.0, -1.0)
    target = ts + tau_us

    bbo = bbo.sort("timestamp")
    b_ts = bbo["timestamp"].to_numpy()
    b_mid = (bbo["bid_price"].to_numpy() + bbo["ask_price"].to_numpy()) * 0.5

    idx = np.searchsorted(b_ts, target, side="right") - 1
    valid = (idx >= 0) & (target >= b_ts[0]) & (target <= b_ts[-1])
    mid = np.where(valid, b_mid[np.clip(idx, 0, len(b_mid) - 1)], np.nan)
    return -sgn * (mid - price) / price * 1e4 + REBATE_BPS


def compute_metrics(
    pnl: np.ndarray, weight: np.ndarray, f: np.ndarray, n_days: float
) -> dict:
    """Score and PnL/turnover figures for one filter at one horizon (spec section 'Score')."""
    valid = np.isfinite(pnl)
    p, w, ff = pnl[valid], weight[valid], f[valid].astype(float)

    keep_w = w * (1.0 - ff)
    filt_w = w * ff
    pnl_all = (w * p).sum() / w.sum()
    pnl_kept = (keep_w * p).sum() / max(keep_w.sum(), 1e-9)
    pnl_filtered = (filt_w * p).sum() / max(filt_w.sum(), 1e-9)
    return dict(
        pnl_all=pnl_all,
        pnl_kept=pnl_kept,
        pnl_filtered=pnl_filtered,
        score=pnl_kept - pnl_all,
        kept_turnover_per_day=keep_w.sum() / n_days,
        kept_frac=(1.0 - ff).mean(),
        n=len(p),
    )


# ----------------------------------------------------------------------------------
# Memory-safe data loading (the full tables are ~14 GB; we load a few full days)
# ----------------------------------------------------------------------------------
DATA = Path(__file__).resolve().parent / "liquidation_task" / "data"
SYM_FILE = {"BTC": "perp_btcusdt", "ETH": "perp_ethusdt"}
BYBIT_FILE = {"BTC": "btcusdt", "ETH": "ethusdt"}


def _epoch_us(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1_000_000)


def _load_window(folder: str, fname: str, lo: int, hi: int, cols: list[str]) -> pl.DataFrame:
    return (
        pl.scan_parquet(DATA / folder / f"{fname}.parquet")
        .filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))
        .select(cols)
        .collect(engine="streaming")
        .sort("timestamp")
    )


DAY_US = 86_400_000_000


def load_day(sym: str, lo: int, hi: int, pad_us: int) -> tuple[pl.DataFrame, ...]:
    """Load one day of trades + bbo (padded for markout) + both liq tables (with lookback pad).

    Loading one day at a time keeps peak memory at ~6M rows so the full month fits a 7 GB box.
    """
    f = SYM_FILE[sym]
    trades = _load_window("binance_trades", f, lo, hi,
                          ["timestamp", "ticker", "side", "price", "amount"])
    bbo = _load_window("binance_booktickers", f, lo, hi + pad_us,
                       ["timestamp", "bid_price", "ask_price"])
    liq_bn = _load_window("binance_liquidations", f, lo - BINANCE_LOOKBACK_US, hi,
                          ["timestamp", "ticker", "side", "price", "amount"])
    liq_by = _load_window("bybit_liquidations", BYBIT_FILE[sym],
                          lo - BYBIT_LOOKBACK_US, hi,
                          ["timestamp", "ticker", "side", "price", "amount"])
    return trades, bbo, liq_bn, liq_by


def _zero_stats() -> dict:
    keys = ["wp", "w", "kwp", "kw", "fwp", "fw", "keep_n", "n"]
    return {tau: {k: 0.0 for k in keys} for tau in TAUS_S}


def _accumulate_day(stats: dict, trades, bbo, liq_bn, liq_by) -> None:
    notional = (trades["price"] * trades["amount"]).to_numpy()
    weight = np.minimum(notional, CLIP)
    filters = classify_trades(trades, bbo, liq_bn, liq_by)
    for tau in TAUS_S:
        pnl = markout_pnl_bps(trades, bbo, tau * 1_000_000)
        valid = np.isfinite(pnl)
        p, w, f = pnl[valid], weight[valid], filters[tau][valid].astype(float)
        keep_w, filt_w = w * (1.0 - f), w * f
        s = stats[tau]
        s["wp"] += float((w * p).sum());        s["w"] += float(w.sum())
        s["kwp"] += float((keep_w * p).sum());  s["kw"] += float(keep_w.sum())
        s["fwp"] += float((filt_w * p).sum());  s["fw"] += float(filt_w.sum())
        s["keep_n"] += float((1.0 - f).sum());  s["n"] += float(len(p))


def evaluate_split(name: str, sym: str, lo: int, hi: int, stride_days: int = 1) -> None:
    """Loop over [lo, hi) one day at a time, accumulate exact weighted sums, then report."""
    pad_us = max(TAUS_S) * 1_000_000 + 5_000_000
    stats = _zero_stats()
    n_days = 0
    day = lo
    while day < hi:
        d_hi = day + DAY_US
        trades, bbo, liq_bn, liq_by = load_day(sym, day, d_hi, pad_us)
        if not trades.is_empty() and not bbo.is_empty():
            _accumulate_day(stats, trades, bbo, liq_bn, liq_by)
            n_days += 1
        day += stride_days * DAY_US

    print(f"\n=== {name}  ({sym}, {n_days} days sampled) ===")
    print(f"  {'tau':>4}  {'Score':>8}  {'PnL_all':>8}  {'PnL_kept':>9}  "
          f"{'PnL_filt':>9}  {'kept%':>6}  {'turn/day(USD)':>15}  {'>=500k?':>8}")
    for tau in TAUS_S:
        s = stats[tau]
        pnl_all = s["wp"] / max(s["w"], 1e-9)
        pnl_kept = s["kwp"] / max(s["kw"], 1e-9)
        pnl_filt = s["fwp"] / max(s["fw"], 1e-9)
        turn_day = s["kw"] / n_days
        kept_frac = s["keep_n"] / max(s["n"], 1e-9)
        ok = "OK" if turn_day >= TURNOVER_MIN else "FAIL"
        print(f"  {tau:>4}  {pnl_kept - pnl_all:>+8.3f}  {pnl_all:>+8.3f}  "
              f"{pnl_kept:>+9.3f}  {pnl_filt:>+9.3f}  {kept_frac*100:>5.1f}%  "
              f"{turn_day:>15,.0f}  {ok:>8}")


def main() -> None:
    sym = "BTC"
    # Sweep every other day across the official splits (train Dec'25-Jan'26, val Feb'26).
    # Day-at-a-time accumulation gives an exact (not sampled-and-scaled) Score & turnover.
    print(f"Baseline ensemble: Bybit-reversal(5s) | Binance-reversal(60s) | cluster>={CLUSTER_MIN}")
    print(f"Turnover constraint: kept clipped turnover >= ${TURNOVER_MIN:,.0f} / day")
    evaluate_split("TRAIN", sym, _epoch_us(2025, 12, 1), _epoch_us(2026, 1, 1), stride_days=2)
    evaluate_split("VAL", sym, _epoch_us(2026, 2, 1), _epoch_us(2026, 3, 1), stride_days=2)


if __name__ == "__main__":
    main()
